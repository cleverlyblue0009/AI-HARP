"""Provenance stamp for every revision-1 output: commit, command line, seeds, hash."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common.config import PROJECT_ROOT, config_hash

OUT_ROOT = PROJECT_ROOT / "results" / "revision1"


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                      text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "--", "agents", "analysis", "common",
                                "configs", "experiments", "hazard", "mobility", "sim"],
                               cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def out_dir(task: str) -> Path:
    d = OUT_ROOT / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(path: Path | str) -> str:
    """Repo-relative path with forward slashes (no absolute local paths in outputs)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def stamp(task: str, config: dict[str, Any], seeds: Any = None, **extra: Any) -> dict[str, Any]:
    """The provenance block written beside every output of ``task``."""
    import platform

    import numpy

    return {"task": task, "git_commit": git_commit(),
            "python": platform.python_version(), "numpy": numpy.__version__,
            "interpreter": _env_name(),
            "command": strip_root(" ".join(["python", "-m", _module()] + sys.argv[1:])),
            "config_hash": config_hash(config), "config": config, "seeds": seeds,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}


def strip_root(text: str) -> str:
    """Replace the absolute project root with '.' (no local paths in committed outputs)."""
    for root in {str(PROJECT_ROOT), str(PROJECT_ROOT.resolve()), PROJECT_ROOT.as_posix()}:
        text = text.replace(root, ".")
    return text


def _env_name() -> str:
    """The environment's folder name (e.g. aiharp-gpu, miniconda3) -- never a full path."""
    p = Path(sys.executable).parent
    return (p.parent if p.name.lower() in ("scripts", "bin") else p).name


def _module() -> str:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    return spec.name if spec is not None else Path(sys.argv[0]).stem


def write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=1, default=_default), encoding="utf-8")
    return path


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)
