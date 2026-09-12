"""YAML config loading, path resolution and content-addressed config hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CACHE_DIR: Path = PROJECT_ROOT / "cache"


def resolve_config_path(name: str | Path) -> Path:
    """Resolve a config reference to a path.

    Accepts an absolute/relative path, a bare stem (``"phy"``) or a filename
    (``"phy.yaml"``); bare names are looked up inside ``configs/``.
    """
    p = Path(name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()
    for candidate in (CONFIG_DIR / p.name, CONFIG_DIR / f"{p.name}.yaml", p):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Config {name!r} not found (looked in {CONFIG_DIR} and as a literal path)."
    )


def load_yaml(name: str | Path) -> dict[str, Any]:
    """Load a YAML config into a plain dict."""
    path = resolve_config_path(name)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping (got {type(data).__name__}).")
    return data


def _canonical(obj: Any) -> Any:
    """Make a config tree JSON-serialisable and order-stable for hashing."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        # Collapse -0.0/0.0 and pin the repr so hashes are stable across runs.
        return f"{obj:.12g}"
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return repr(obj)


def config_hash(*objs: Any, length: int = 12) -> str:
    """Short stable hash over one or more config trees.

    Every row in ``results/runs.csv`` carries this so a run is never silently
    regenerated from a different configuration.
    """
    payload = json.dumps([_canonical(o) for o in objs], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
