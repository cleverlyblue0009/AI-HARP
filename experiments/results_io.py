"""Append-only results store.

Every run writes one row to ``results/runs.csv`` keyed by its ``config_hash``,
so a completed run is never silently recomputed with a different configuration
and never duplicated. This is the single source of truth that Phase 7's
aggregation and Phase 8's figures read from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common.config import RESULTS_DIR, ensure_dir
from common.logging_utils import get_logger

logger = get_logger("experiments.results")

RUNS_CSV = RESULTS_DIR / "runs.csv"

#: Identity of a run. Two rows agreeing on all of these are the same run.
#:
#: `metrics_version` is part of the key, not just a label: the same config run
#: under a changed metric definition is a DIFFERENT result, and de-duplicating
#: on config alone would keep the stale row and silently discard the new one.
KEY_COLUMNS = ("config_hash", "metrics_version")


def load_runs(path: Path = RUNS_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def append_runs(
    rows: Iterable[dict[str, Any]], path: Path = RUNS_CSV, *, overwrite: bool = False
) -> pd.DataFrame:
    """Append rows, de-duplicating on ``config_hash``.

    With ``overwrite=False`` (the default) an existing row wins, so re-running a
    sweep is idempotent and cheap. With ``overwrite=True`` the new row replaces
    it -- use that after deliberately changing a config.
    """
    new = pd.DataFrame(list(rows))
    if new.empty:
        return load_runs(path)

    missing = [c for c in KEY_COLUMNS if c not in new.columns]
    if missing:
        raise ValueError(f"Result rows are missing key column(s): {missing}")

    old_all = load_runs(path)
    if not old_all.empty and "metrics_version" in old_all.columns:
        stale = old_all[old_all["metrics_version"] != new["metrics_version"].iloc[0]]
        if len(stale):
            logger.warning(
                "%d existing row(s) carry a different metrics_version; they are "
                "kept but must not be mixed with these in a figure or table.",
                len(stale),
            )
    elif not old_all.empty:
        logger.warning(
            "%d existing row(s) predate metrics versioning (metrics_version < %d) "
            "and are NOT comparable with current rows.", len(old_all), 2,
        )

    old = load_runs(path)
    if old.empty:
        combined = new
    else:
        combined = pd.concat([old, new], ignore_index=True)
        keep = "last" if overwrite else "first"
        before = len(combined)
        combined = combined.drop_duplicates(subset=list(KEY_COLUMNS), keep=keep)
        dropped = before - len(combined)
        if dropped:
            logger.info("Skipped %d already-recorded run(s)", dropped)

    ensure_dir(path.parent)
    combined.to_csv(path, index=False)
    logger.info("results: %d rows -> %s", len(combined), path)
    return combined


def already_done(config_hash: str, path: Path = RUNS_CSV) -> bool:
    df = load_runs(path)
    return not df.empty and (df["config_hash"] == config_hash).any()
