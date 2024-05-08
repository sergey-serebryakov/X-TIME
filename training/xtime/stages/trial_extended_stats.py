###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import click
import typing as t
from multiprocessing import Process
from pathlib import Path
from pprint import pprint
import numpy as np
import pandas as pd
import mlflow
from mlflow.entities import Run
from mlflow.utils.file_utils import local_file_uri_to_path
import random

from xtime.contrib.tune_ext import get_trial_dir
from xtime.datasets import Dataset
from tqdm import tqdm

from xtime.io import IO
from xtime.ml import Task
from xtime.run import RunType
from dataclasses import dataclass, field


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


@dataclass
class WorkerConfig:
    cpu_affinity: t.List[int] = field(default_factory=list)
    num_trials: int = -1
    shuffle_trials: bool = False
    trial_dirs: t.List[Path] = field(default_factory=list)
    output_file: str = ""
    backup_trial_dir_resolver: t.Optional[t.Callable] = None
    model_min: int = 1
    model_max: int = -1
    model_step: int = 100


def get_trial_extended_stats(run: str, dataset: t.Optional[Dataset], config: WorkerConfig) -> t.List[t.Dict]:
    """
    kwargs:
        cpu_affinity (list[int])
        num_trials (int), shuffle_trials (bool)
        trial_dirs: list[Path]
        output_file (str)
        backup_trial_dir_resolver (callable)
    """
    if config.cpu_affinity:
        import psutil
        myself = psutil.Process()
        myself.cpu_affinity(config.cpu_affinity)

    from xtime.estimators.estimator import Model

    mlflow_run = mlflow.get_run(run)
    run_type: RunType = RunType(mlflow_run.data.tags["run_type"])
    if run_type != RunType.HPO:
        raise ValueError(f"Unsupported MLflow run ({mlflow_run.data.tags['run_type']})")
    trials_stats: t.List[t.Dict] = []
    artifact_path: Path = Path(local_file_uri_to_path(mlflow_run.info.artifact_uri))
    task: Task = Task.from_dataset_info(IO.load_yaml(artifact_path / "dataset_info.yaml"))

    if dataset is None:
        dataset = Dataset.create(mlflow_run.data.tags["dataset_name"])

    if not config.trial_dirs:
        trial_dirs: t.List[Path] = [_dir for _dir in (artifact_path / "ray_tune").iterdir() if _dir.is_dir()]
    else:
        trial_dirs = config.trial_dirs
        assert isinstance(trial_dirs, list)

    if 0 < config.num_trials < len(trial_dirs):
        if config.shuffle_trials:
            random.shuffle(trial_dirs)
        trial_dirs = trial_dirs[0: config.num_trials]

    if mlflow_run.data.tags["model"] == "xgboost":
        from xtime.contrib.xgboost_ext import TreeModel
        ...
    elif mlflow_run.data.tags["model"] in ("rf", "rf_clf"):
        raise ValueError(f"Not implemented yet")
        # from xtime.contrib.sklearn_ext import get_model_stats
        # ...
    else:
        raise ValueError(f"Unsupported model ({mlflow_run.data.tags['model']})")

    trail_common_stats = {
        "model_name": mlflow_run.data.tags["model"],
        "dataset_name": mlflow_run.data.tags["problem"] if "problem" in mlflow_run.data.tags else
        mlflow_run.data.tags["dataset_name"],
        "tune_root_path": (artifact_path / "ray_tune").as_posix(),
        "mlflow_run_id": mlflow_run.info.run_id,
    }

    _known_metrics = {
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

    trial_index: int = 0
    for trial_dir in tqdm(
            trial_dirs, total=len(trial_dirs), desc="Hyperparameter search trials", unit="trials", leave=False
    ):
        model_file: str = Model.get_file_name(mlflow_run.data.tags["model"])
        resolved_trial_dir = get_trial_dir(trial_dir, model_file, config.backup_trial_dir_resolver)
        if resolved_trial_dir is None:
            print(f"WARNING no valid trial directory found (id={mlflow_run.info.run_id}, path={trial_dir}).")
            continue
        trial_stats = trail_common_stats.copy()
        trial_stats.update({
            "model_file": (resolved_trial_dir / model_file).as_posix(),
            "tune_trial_path": resolved_trial_dir.as_posix(),
        })

        model_params: t.Dict = IO.load_dict(resolved_trial_dir / "params.json")
        for k, v in model_params.items():
            trial_stats["param_" + k] = v

        tree_model = TreeModel(Model.load_model(resolved_trial_dir, "xgboost", task.type))

        num_models = min(
            tree_model.ensemble_size,
            tree_model.ensemble_size if config.model_max < 0 else config.model_max
        )
        rounds_under_test = list(range(config.model_min, num_models, config.model_step))
        if len(rounds_under_test) > 0 and rounds_under_test[-1] != num_models:
            rounds_under_test.append(num_models)

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
            metrics = tree_model.estimator.evaluate(
                dataset, predict_proba_kwargs={"iteration_range": (0, round_under_test)}
            )
            for k, v in metrics.items():
                if k in _known_metrics:
                    round_stats["metric_" + k] = v

            trials_stats.append(round_stats)

        trial_index += 1
        if config.output_file and trial_index % 10 == 0:
            pd.DataFrame(trials_stats).to_csv(config.output_file, index=False)

    if config.output_file:
        pd.DataFrame(trials_stats).to_csv(config.output_file, index=False)
        print("done processing:", config.output_file)
    return trials_stats


def parallel(
        run: Run, output_dir: Path, num_workers: int, num_worker_cores: int, version: str,
        model_min: int, model_max: int, model_step: int
) -> None:
    artifact_path: Path = Path(local_file_uri_to_path(run.info.artifact_uri))
    trial_dirs: t.List[Path] = [_dir for _dir in (artifact_path / "ray_tune").iterdir() if _dir.is_dir()]
    random.shuffle(trial_dirs)

    chunks: t.List[np.ndarray] = np.array_split(np.array(trial_dirs), num_workers)
    assert isinstance(chunks, list)

    model: str = run.data.tags["model"]
    dataset_name: str = run.data.tags["dataset_name"]

    processes = []
    partial_results = []
    for idx, chunk in enumerate(chunks):
        partial_results.append(f"{version}_{model}_{dataset_name}_{model_min}_{model_step}_{model_max}_{idx}.csv.gz")
        p = Process(
            target=get_trial_extended_stats,
            args=(run.info.run_id,),
            kwargs=dict(
                dataset=None,
                config=WorkerConfig(
                    num_trials=-1,
                    trial_dirs=chunk.tolist(),
                    output_file=(output_dir / partial_results[-1]).as_posix(),
                    cpu_affinity=list(range(num_worker_cores * idx, num_worker_cores * (idx + 1))),
                    model_min=model_min,
                    model_max=model_max,
                    model_step=model_step
                )
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes finished")


@click.command()
@click.option('--run', 'run_id', required=True, type=str, metavar="MLFLOW_RUN_ID")
@click.option('--output-dir', required=True, type=str, metavar="OUTPUT_DIRECTORY")
@click.option('--num-workers', required=False, type=int, default=1)
@click.option('--num-worker-cores', required=False, type=int, default=-1)
@click.option('--version', required=True, type=str)
@click.option('--model-min', required=False, type=int, default=1)
@click.option('--model-max', required=False, type=int, default=-1)
@click.option('--model-step', required=False, type=int, default=100)
def main(
        run_id: str, output_dir: str, num_workers: int, num_worker_cores: int, version: str, model_min: int,
        model_max: int, model_step: int
) -> None:
    if num_worker_cores <= 0:
        import psutil
        num_worker_cores = max(1, psutil.cpu_count(logical=True) // num_workers)

    print(f"run_id={run_id}, num_workers={num_workers}, num_worker_cores={num_worker_cores}")

    run: Run = mlflow.get_run(run_id)
    pprint(run.data.tags)

    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    parallel(run, _output_dir, num_workers, num_worker_cores, version, model_min, model_max, model_step)


if __name__ == "__main__":
    main()
