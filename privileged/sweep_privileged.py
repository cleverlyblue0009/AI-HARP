"""Revision-5 protocol sweep for a rung-C (ORACLE_INPUT) checkpoint: experiments.revision5.sweep with the
oracle-relevance node input supplied. Non-deployable upper bound; see privileged/__init__.py.
::
    D:/aiharp-gpu/Scripts/python.exe -m privileged.sweep_privileged agent --seedset SEL --cells headline \
        --runs rungC_s0 rungC_s1 --tag ladderC --jobs 5

The patch is installed at import (as privileged.eval_privileged), so the runner's spawned workers, which
re-import this __main__ module, get it too.
"""
from __future__ import annotations

import privileged.eval_privileged  # noqa: F401  (installs the oracle-input risk-field builder)


def main(argv=None) -> int:
    from experiments.revision5.sweep import main as sweep_main
    return sweep_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
