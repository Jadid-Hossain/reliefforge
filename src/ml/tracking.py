"""MLflow tracking helpers (course requirement: >= 2 runs with different settings).

mlflow >= 3 note: `mlflow.start_run()` returns an `ActiveRun` WITHOUT log_* methods.
All logging therefore goes through the module-level fluent functions, which write to
the ACTIVE run. The `run` arguments are kept for call-site compatibility.
"""
from __future__ import annotations

import os
from pathlib import Path


def is_active() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except Exception:
        return False


def start_run(experiment: str, run_name: str, tracking_uri: str | None = None, tags: dict | None = None):
    import mlflow

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
    """Log a (nested) config as flat params to the active run. Missing keys never crash training."""
    import logging

    import mlflow

    log = logging.getLogger("reliefforge.mlflow")
    for k, v in _flatten(cfg).items():
        val = str(v)
        if len(val) > 400:
            val = val[:397] + "..."
        try:
            mlflow.log_param(k, val)
        except Exception as exc:  # visible: a lost param is a reproducibility bug
            log.warning("mlflow log_param(%s) failed: %s", k, exc)


def log_metrics_safe(run, metrics: dict, step: int) -> None:
    import logging

    import mlflow

    log = logging.getLogger("reliefforge.mlflow")
    for k, v in metrics.items():
        if v is None or v != v:  # None or NaN
            continue
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception as exc:
            log.warning("mlflow log_metric(%s) failed: %s", k, exc)


def log_artifact_safe(run, path, artifact_path=None) -> None:
    import logging

    import mlflow

    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        logging.getLogger("reliefforge.mlflow").warning("mlflow log_artifact(%s) failed: %s", path, exc)


def log_dict_json(run, data: dict, artifact_file: str) -> None:
    import logging

    import mlflow

    try:
        mlflow.log_dict(run, data, artifact_file)
    except Exception as exc:
        logging.getLogger("reliefforge.mlflow").warning("mlflow log_dict(%s) failed: %s", artifact_file, exc)


def end_run(status: str = "FINISHED") -> None:
    import mlflow

    try:
        mlflow.end_run(status=status)
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
