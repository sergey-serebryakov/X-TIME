###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
import logging
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###
import os
import sys
import time
import math
import click
import json
import typing as t
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import pandas as pd
import mlflow
from mlflow.entities import Run
import random

from xtime import hparams
from xtime.contrib.mlflow_ext import MLflow
from xtime.contrib.tune_ext import get_trial_dir
from xtime.datasets import Dataset
from xtime.estimators.estimator import Model
from xtime.io import IO
from xtime.run import RunType
from dataclasses import dataclass, field
import queue
import coloredlogs


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

logger: t.Optional[logging.Logger] = None


class Event:
    def __init__(self, subsystem: str, message: t.Optional[str] = None, **kwargs) -> None:
        self.subsystem = subsystem
        self.message = message
        self.kwargs = kwargs

    def __str__(self) -> str:
        if self.message:
            return json.dumps({"subsystem": self.subsystem, "message": self.message, "args": self.kwargs})
        return json.dumps({"subsystem": self.subsystem, "args": self.kwargs})


def get_number_of_gpus() -> int:
    """Return number of available GPUs."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount()
    except ImportError:
        raise ImportError("Cupy is not installed. Please install it to detect the number of GPUs.")


class RunningStats:
    """https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation"""

    def __init__(self) -> None:
        self.n: int = 0
        self.old_m: float = 0
        self.new_m: float = 0
        self.old_s: float = 0
        self.new_s: float = 0

    def reset(self) -> None:
        self.n = 0

    def push(self, x: float) -> None:
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self) -> float:
        return self.new_m if self.n else 0.0

    def var(self) -> float:
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self) -> float:
        return math.sqrt(self.var())


@dataclass
class ModelsConfig:
    min_: int = 1
    max_: int = -1
    step: int = 100

    def get_models(self, ensemble_size: int) -> t.List[int]:
        num_models = min(
            ensemble_size,
            ensemble_size if self.max_ < 0 else self.max_
        )
        models_under_test = list(range(self.min_, num_models, self.step))
        if len(models_under_test) > 0 and models_under_test[-1] != num_models:
            models_under_test.append(num_models)
        return models_under_test


@dataclass
class WorkerConfig:
    cpu_affinity: t.List[int] = field(default_factory=list)
    env: t.Dict = field(default_factory=dict)
    num_trials: int = -1
    shuffle_trials: bool = False
    output_file: str = ""
    backup_trial_dir_resolver: t.Optional[t.Callable] = None
    models_config: ModelsConfig = field(default_factory=ModelsConfig)
    dataset_splits: str = ""

    def configure_process(self) -> None:
        if self.cpu_affinity:
            import psutil
            myself = psutil.Process()
            myself.cpu_affinity(self.cpu_affinity)

        if self.env:
            for name, value in self.env.items():
                os.environ[name] = value


def get_mlflow_run(run_id: str) -> Run:
    run = mlflow.get_run(run_id)
    run_type: RunType = RunType(run.data.tags["run_type"])
    if run_type != RunType.HPO:
        raise ValueError(f"[worker] [{os.getpid()}] unsupported MLflow run ({run.data.tags['run_type']}).")
    return run


def get_dataset(dataset: t.Optional[Dataset], run: Run, config: WorkerConfig) -> Dataset:
    if dataset is None:
        dataset = Dataset.create(run.data.tags["dataset_name"] + ":" + run.data.tags["dataset_version"])
    if config.dataset_splits:
        splits_to_keep = config.dataset_splits.strip().split(",")
        splits_to_remove = [name for name in dataset.splits.keys() if name not in splits_to_keep]
        for split_to_remove in splits_to_remove:
            del dataset.splits[split_to_remove]
        print(
            f"[worker] [{os.getpid()}] [get_dataset] splits_to_keep={splits_to_keep}, "
            f"splits_to_remove={splits_to_remove}, splits_kept={list(dataset.splits.keys())}."
        )
    for split_name, split in dataset.splits.items():
        print(f"[worker] [{os.getpid()}] [get_dataset] split={split_name}, shape={split.y.shape}.")
    return dataset


class _Queue:
    def __init__(self, trial_dirs: t.List[Path]) -> None:
        self._trial_dirs = [trial_dir.as_posix() for trial_dir in trial_dirs] + [None]
        self._next = 0

    def get(self, *args, **kwargs) -> t.Optional[str]:
        if self._next < len(self._trial_dirs):
            trial_dir = self._trial_dirs[self._next]
            self._next += 1
            return trial_dir
        raise queue.Empty


def save_trials_stats(
        trials_stats: t.List[t.Dict], config: WorkerConfig,
        num_trials: t.Optional[int] = None, done: t.Optional[bool] = None
) -> None:
    if not config.output_file:
        return

    if (done is not None and done is True) or (num_trials is not None and num_trials % 10 == 0):
        pd.DataFrame(trials_stats).to_csv(config.output_file, index=False)


def get_expected_metrics(dataset: Dataset) -> t.Set[str]:
    if dataset.metadata.task.type.classification():
        return {
            "dataset_accuracy",
            "dataset_loss_total",
            "train_accuracy",
            "train_loss_mean",
            "train_loss_total",
            "valid_accuracy",
            "valid_loss_mean",
            "valid_loss_total",
            "test_accuracy",
            "test_loss_mean",
            "test_loss_total",
            "dataset_loss_mean",
        }
    return {
        "dataset_mse", "test_mse", "test_r2", "test_r2_oos", "train_mse", "train_r2", "train_r2_oos", "valid_mse"
    }


def set_log_level(log_level: str) -> None:
    log_level = log_level.upper()
    logging.basicConfig(level=log_level, force=True)
    coloredlogs.install(level=log_level)


def get_trial_extended_stats(
        run_id: str, dataset: t.Optional[Dataset], task_queue: t.Union[mp.Queue, _Queue], config: WorkerConfig
) -> None:
    config.configure_process()

    run = get_mlflow_run(run_id)
    dataset = get_dataset(dataset, run, config)

    artifact_path: Path = MLflow.get_artifact_path(run)
    trials_stats: t.List[t.Dict] = []

    if run.data.tags["model"] in {"xgboost", "xgboost-rf"}:
        from xtime.contrib.xgboost_ext import TreeModel
        ...
    elif run.data.tags["model"] == "rapids-rf":
        from xtime.contrib.rapids_ext import TreeModel
        ...
    elif run.data.tags["model"] in {"rf", "rf_clf"}:
        raise ValueError(f"Not implemented yet")
    else:
        raise ValueError(f"Unsupported model ({run.data.tags['model']})")

    trail_common_stats = {
        "model_name": run.data.tags["model"],
        "dataset_name": run.data.tags["problem"] if "problem" in run.data.tags else
        run.data.tags["dataset_name"],
        "tune_root_path": (artifact_path / "ray_tune").as_posix(),
        "mlflow_run_id": run.info.run_id,
    }
    _expected_metrics = get_expected_metrics(dataset)

    num_trials = 0
    rs = RunningStats()
    while True:
        st_tm = time.time()
        try:
            _trial_dir: str = task_queue.get(block=True, timeout=0.1)
            if _trial_dir is None:
                break
            trial_dir = Path(_trial_dir)
        except queue.Empty:
            continue
        num_trials += 1
        model_file: str = Model.get_file_name(run.data.tags["model"])
        resolved_trial_dir = get_trial_dir(trial_dir, model_file, config.backup_trial_dir_resolver)
        if resolved_trial_dir is None:
            print(f"WARNING no valid trial directory found (id={run.info.run_id}, path={trial_dir}).")
            continue
        trial_stats = trail_common_stats.copy()
        trial_stats.update({
            "model_file": (resolved_trial_dir / model_file).as_posix(),
            "tune_trial_path": resolved_trial_dir.as_posix(),
        })

        model_params: t.Dict = IO.load_dict(resolved_trial_dir / "params.json")
        for k, v in model_params.items():
            trial_stats["param_" + k] = v

        tree_model = TreeModel(Model.load_model(resolved_trial_dir))
        rounds_under_test: t.List[int] = config.models_config.get_models(tree_model.ensemble_size)
        for round_under_test in rounds_under_test:
            round_stats = trial_stats.copy()
            try:
                model_stats: t.Dict = tree_model.describe(round_under_test)
                for k, v in model_stats.items():
                    round_stats["model_" + k] = v
            except (IOError, EOFError):
                # When there's something wrong with the serialized model.
                print("WARNING can't get model stats, dir=", resolved_trial_dir.as_posix())
                continue

            #
            metrics = tree_model.evaluate(dataset, round_under_test)
            for k, v in metrics.items():
                if k in _expected_metrics:
                    round_stats["metric_" + k] = v

            trials_stats.append(round_stats)

        save_trials_stats(trials_stats, config, num_trials=num_trials)
        rs.push(time.time() - st_tm)
        print(f"[worker] [{os.getpid()}] num_trials={num_trials}, trial_tm={rs.mean()}, trial_std={rs.std()}.")
        if 0 < config.num_trials <= num_trials:
            break

    save_trials_stats(trials_stats, config, done=True)
    print(f"done processing: pid={os.getpid()}, output_file={config.output_file}, num_trials={num_trials}")


class Driver:
    @staticmethod
    def run(
            run: Run, output_dir: Path, num_workers: int, num_gpus: int, cores_per_worker: int,
            version: str, models_config: ModelsConfig, dataset_splits: str
    ) -> None:
        start_tm = time.time()
        artifact_path: Path = MLflow.get_artifact_path(run)
        if not artifact_path.is_dir():
            logger.error(
                Event("init", "artifact directory not found", artifact_path=artifact_path.as_posix())
            )
            return

        trial_dirs: t.List[Path] = [_dir for _dir in (artifact_path / "ray_tune").iterdir() if _dir.is_dir()]
        if not trial_dirs:
            logger.error(
                Event("init", "trials not found", tune_dir=(artifact_path / "ray_tune").as_posix())
            )
            return
        random.shuffle(trial_dirs)
        logger.info(Event("init", num_trials=len(trial_dirs)))

        model: str = run.data.tags["model"]
        dataset_name: str = run.data.tags["dataset_name"]
        print(f"[driver] [{os.getpid()}] [parallel] model={model}, dataset_name={dataset_name}")

        output_file: str = f"{version}_{model}_{dataset_name}_{models_config.min_}_{models_config.max_}_{models_config.step}.csv.gz"
        if num_workers == 1:
            get_trial_extended_stats(
                run.info.run_id,
                dataset=None,
                task_queue=_Queue(trial_dirs),
                config=WorkerConfig(
                    num_trials=-1,
                    output_file=(output_dir / output_file).as_posix(),
                    cpu_affinity=list(range(cores_per_worker)) if cores_per_worker > 0 else [],
                    env={"CUDA_VISIBLE_DEVICES": "" if num_gpus == 0 else str(0)},
                    models_config=models_config,
                    dataset_splits=dataset_splits.strip() if isinstance(dataset_splits, str) else ""
                )
            )
        else:
            processes = []
            partial_results = []
            task_queue = mp.Queue()
            for idx in range(num_workers):
                partial_results.append(
                    f"{version}_{model}_{dataset_name}_{models_config.min_}_{models_config.max_}_{models_config.step}_{idx}.csv.gz"
                )

                cpu_affinity = []
                if cores_per_worker > 0:
                    cpu_affinity = list(range(cores_per_worker * idx, cores_per_worker * (idx + 1)))

                env = {
                    "CUDA_VISIBLE_DEVICES": "" if num_gpus == 0 else str(idx % num_gpus)
                }

                p = mp.Process(
                    target=get_trial_extended_stats,
                    args=(run.info.run_id,),
                    kwargs=dict(
                        dataset=None,
                        task_queue=task_queue,
                        config=WorkerConfig(
                            num_trials=-1,
                            output_file=(output_dir / partial_results[-1]).as_posix(),
                            cpu_affinity=cpu_affinity,
                            env=env,
                            models_config=models_config,
                            dataset_splits=dataset_splits.strip() if isinstance(dataset_splits, str) else ""
                        )
                    )
                )
                processes.append(p)
                p.start()

            for trial_dir in trial_dirs:
                task_queue.put(trial_dir.as_posix())
            for _ in range(num_workers):
                task_queue.put(None)

            # Indicate no more data will be put on this queue and wait for data to be flushed to the pipe.
            task_queue.close()
            task_queue.join_thread()

            # Wait for workers to process all submitted tasks.
            for p in processes:
                p.join()

        current_tm = time.time()
        print(f"All processes finished in {current_tm - start_tm} seconds.")


@click.command()
@click.option(
    "--log-level",
    "--log_level",
    required=False,
    default="debug",
    type=click.Choice(["critical", "error", "warning", "info", "debug"]),
    help="Logging level is a lower-case string value for Python's logging library (see "
         "[Logging Levels]({log_level}) for more details). Only messages with this logging level or higher are "
         "logged.".format(log_level="https://docs.python.org/3/library/logging.html#logging-levels"),
)
@click.option(
    '--run', required=True, type=str, metavar="MLFLOW_RUN_ID",
    callback=lambda ctx, param, value: mlflow.get_run(value),
    help="MLflow run ID with hyperparameter search results for tree-based models."
)
@click.option(
    '--output-dir', required=True, type=str, metavar="OUTPUT_DIRECTORY",
    callback=lambda ctx, param, value: Path(value),
    help="Output directory to store results."
)
@click.option(
    '--num-workers', required=False, type=int, default=1,
    help="Number of workers to use for parallel processing."
)
@click.option(
    '--num-gpus', required=False, type=int, default=0,
    callback=lambda ctx, param, value: value if value >= 0 else get_number_of_gpus(),
    help="Total number of available GPUs. 0: do not use GPUs, -1: detect automatically."
)
@click.option(
    '--cores-per-worker', required=False, type=int, default=-1,
    help="Number of CPU cores per one worker. 0: do not set affinity, -1: detect automatically."
)
@click.option(
    '--version', required=True, type=str,
    help="A version string that will become of the output file name."
)
@click.option(
    '--trees', required=False, type=str, default="min=1;max=-1;step=100",
    callback=lambda ctx, param, value: ModelsConfig(**hparams.from_string(value)),
    help="Specification for number of trees to explore."
)
@click.option(
    '--dataset-splits', required=False, type=str, default="",
    help="List of dataset splits to use. If not present or empty, use all splits defined in the dataset."
)
@click.option("--init-run", is_flag=True)
def main(
        run: Run, output_dir: Path, num_workers: int, num_gpus: int, cores_per_worker: int, version: str,
        trees: ModelsConfig, dataset_splits: str, log_level: str, init_run: bool
) -> None:
    global logger
    set_log_level(log_level)
    logger = logging.getLogger("driver")

    logger.info(Event("init", unparsed_args=sys.argv), extra={"bla": "kva"})

    output_dir.mkdir(parents=True, exist_ok=True)
    if cores_per_worker < 0:
        import psutil
        cores_per_worker = max(1, psutil.cpu_count(logical=True) // num_workers)

    logger.info(Event(
        "init",
        parased_args={
            "run_id": run.info.run_id, "output_dir": output_dir.as_posix(), "num_workers": num_workers,
            "num_gpus": num_gpus, "cores_per_worker": cores_per_worker, "version": version, "trees": str(trees),
            "dataset_splits": dataset_splits
        }
    ))
    logger.info(Event("init", run_tags=run.data.tags))

    if init_run:
        print("source run")
        print(f"\trun_id = {run.info.run_id}")
        print(f"\trun_name = {run.info.run_name}")
        print(
            "\tdataset ={name}:{version}".format(
                name=run.data.tags["dataset_name"], version=run.data.tags["dataset_version"]
            )
        )
        print(f"\tmodel = {run.data.tags['model']}")
        print(f"\tmlflow.note.content = {run.data.tags['mlflow.note.content']}")
        return

    Driver().run(run, output_dir, num_workers, num_gpus, cores_per_worker, version, trees, dataset_splits)


if __name__ == "__main__":
    main()
