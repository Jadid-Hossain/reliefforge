"""MLflow tracking helpers (course requirement: >= 2 runs with different settings).

The trainer calls these; if MLflow cannot even be imported the calls degrade to
no-ops so the training itself never breaks.
"""
from __future__ import annotations

import os
from pathlib import Path


class _NullRun:
    def __setitem__(self, k, v):
        pass

    def log_artifacts(self, *a, **k):
        pass


def is_active() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except Exception:
        return False


def start_run(experiment: str, run_name: str, tracking_uri: str | None = None, tags: dict | None = None):
    import mlflow
    from pathlib import Path

    # mlflow >= 3.18 keeps the file store behind an opt-out flag; we deliberately use it:
    # mlruns/ is plain files that can be committed as course evidence.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = (Path.cwd() / "mlruns").resolve().as_uri()  # file store -> committed to the repo
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment)
    except Exception as exc:
        print(f"mlflow: could not set experiment '{experiment}': {exc}")
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def log_params_flat(run, cfg: dict, prefix: str = "") -> None:
    for k, v in _flatten(cfg).items():
        val = str(v)
        if len(val) > 400:
            val = val[:397] + "..."
        try:
            run.log_param(k, val)
        except Exception:
            pass


def log_metrics_safe(run, metrics: dict, step: int) -> None:
    for k, v in metrics.items():
        if v is None or v != v:  # None or NaN
            continue
        try:
            run.log_metric(k, float(v), step=step)
        except Exception:
            pass


def log_artifact_safe(run, path, artifact_path=None) -> None:
    try:
        run.log_artifact(str(path), artifact_path=artifact_path)
    except Exception:
        pass


def log_dict_json(run, data: dict, artifact_file: str) -> None:
    try:
        import json

        import mlflow

        mlflow.log_dict(run, data, artifact_file)
    except Exception:
        pass


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
