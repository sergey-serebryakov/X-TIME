# Script Dependencies:
#    mlflow
#    fire

import contextlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import typing as t
from dataclasses import dataclass
from pathlib import Path

import fire
import mlflow
import mlflow.entities
from mlflow.entities import RunStatus
from mlflow.utils.file_utils import local_file_uri_to_path


@dataclass
class Args:
    name: str
    """MLflow experiment name where NOX session runs are logged."""
    rm: bool
    """If true, remove python environment after executing a nox session."""
    rerun: bool
    """Rerun a NOX session even if its previous execution was successful."""


def run_nox_sessions(
    *sessions: str, name: str = "nox", rm: bool = True, rerun: bool = False, backend: t.Literal["default", "ray"]
) -> None:
    """Run multiple NOX sessions sequentially logging results with MLflow.

    Args:
        sessions: List of NOX sessions to run. If not provided, all NOX sessions will run defined in `noxfile.py`.
        name: MLflow experiment name under which MLflow runs will be created (one run per one NOX session).
        rm: Remove python environment after executing a nox session. Helps to save space.
        rerun: By default, only failed NOX sessions are rerun. If this flag is set, all sessions are rerun.
        backend: Backend to run NOX sessions. Should be either `default` or `ray`.
    """
    print("sessions:", sessions, ", name:", name, ", rm", rm, ", rerun:", rerun)
    assert name, "The --name flag must hold a non-empty string value."
    assert sessions, "The session flag must hold a non-empty list of sessions."
    assert backend in {"default", "ray"}, f"Invalid backend ({backend})."

    args = Args(name=name, rm=rm, rerun=rerun)

    if backend == "default":
        # Run sessions sequentially in this process.
        with working_directory(Path(__file__).resolve().parent):
            for session in sessions:
                run_nox_session(session, args)
    else:
        print(f"Unsupported backend ({backend}).", file=sys.stderr)
        exit(1)


def run_nox_session(session: str, args: Args) -> None:
    """Run one NOX session.

    Args:
        session: NOX session name.
        args: Command line arguments.
    """
    experiment_id: str = _get_experiment_id(args.name, create_if_not_exist=True)
    assert experiment_id is not None, "Internal bug (experiment ID cannot be none)."

    # Transform this `model_train_test-3.9(model='xgboost')` to model_train_test-3.9-model-xgboost`
    run_name = session.replace("'", "").replace('"', "").replace("=", "-").replace("(", "-").replace(")", "")

    runs: t.List[mlflow.entities.Run] = mlflow.search_runs(
        [experiment_id], filter_string=f"run_name='{run_name}'", output_format="list"
    )
    assert len(runs) <= 1, "Internal bug (more than one run with the same name found)."

    if not _should_run(runs, args):
        print(f"Skipping session (name={session}).")
        return

    run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    mlflow.log_param("session", session)
    status = RunStatus.FINISHED

    artifact_dir = Path(local_file_uri_to_path(run.info.artifact_uri))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stdout, stderr = open(artifact_dir / "stdout.txt", "wt"), open(artifact_dir / "stderr.txt", "wt")

    env_dir = artifact_dir / "nox"
    report_file = artifact_dir / "report.json"

    try:
        cmd = [
            "nox",
            "--session",
            session,
            "--envdir",
            env_dir.as_posix(),
            "--verbose",
            "--report",
            report_file.as_posix(),
        ]
        stdout.write("Nox command: " + " ".join(cmd) + "\n")

        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        _ = process.communicate()
        if process.returncode != 0:
            status = RunStatus.FAILED

        stdout.write(f"Processing report file ({report_file}).\n")
        if report_file.is_file():
            with open(report_file, "rt") as file:
                report = json.load(file)
                mlflow.log_metrics({"result_code": int(report["sessions"][0]["result_code"])})
                mlflow.set_tag("name", report["sessions"][0]["name"])
                if report["sessions"][0]["result"] != "success":
                    status = RunStatus.FAILED
        else:
            stderr.write(f"No report file found ({report_file}).\n")
            status = RunStatus.FAILED
    except Exception as err:
        stderr.write(str(err) + "\n")
        stderr.write(traceback.format_exc() + "\n")
        status = RunStatus.FAILED
    finally:
        if args.rm is True and env_dir.is_dir():
            stdout.write(f"Removing python environment directory ({env_dir}).\n")
            shutil.rmtree(env_dir)

        stdout.close()
        stderr.close()

        mlflow.end_run(status=RunStatus.to_string(status))


@contextlib.contextmanager
def working_directory(work_dir: Path) -> Path:
    """Temporarily change current working directory.

    Args:
        work_dir: Working directory to set.
    Returns:
        The input `work_dir` parameter.
    """
    cur_work_dir = Path.cwd().resolve()
    os.chdir(work_dir)

    try:
        yield work_dir
    finally:
        os.chdir(cur_work_dir)


def _should_run(runs: t.List[mlflow.entities.Run], args: Args) -> bool:
    """Determine if this session needs to run.

    Args:
        runs: Possibly non-empty list of MLflow runs that correspond to the given test session. Should always contain
            zero or one entry.
        args: Command line arguments. We need the `rerun` flag there.
    """
    assert len(runs) <= 1, "Internal bug (function expects at least one run)."

    if not runs:
        return True  # A session has not run yet.

    run = runs[0]
    status = RunStatus.from_string(run.info.status)

    if status in (RunStatus.RUNNING, RunStatus.SCHEDULED):  # Session run in progress or has been scheduled.
        return False

    if status in (RunStatus.FAILED, RunStatus.KILLED):  # Previous session run failed to complete.
        mlflow.delete_run(run.info.run_id)
        return True

    if status == RunStatus.FINISHED:  # Previous run has completed successfully.
        if not args.rerun:
            return False
        mlflow.delete_run(runs[0].info.run_id)  # But arguments can specify to always rerun.
        return True

    print(f"Unexpected run status ({run.info.status}). Will rerun just in case.")
    mlflow.delete_run(run.info.run_id)
    return True


def _get_experiment_id(experiment_name: str, create_if_not_exist: bool = True) -> t.Optional[str]:
    """Return experiment ID by the experiment name.

    Args:
        experiment_name: Name of the MLflow experiment.
        create_if_not_exist: If True, create the experiment if it does not exist.
    Returns:
        Experiment ID. If experiment does not exist, and `create_if_not_exist` is False, None is returned.
    """
    experiment: t.Optional[mlflow.entities.Experiment] = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    if not create_if_not_exist:
        return None
    return mlflow.create_experiment(experiment_name)


if __name__ == "__main__":
    fire.Fire(run_nox_sessions)
