"""Evaluate a rung-C (ORACLE_INPUT) checkpoint: experiments.revision1.agent_eval with the
oracle-relevance node input supplied. Non-deployable upper bound; see privileged/__init__.py.

::

    D:/aiharp-gpu/Scripts/python.exe -m privileged.eval_privileged --checkpoint checkpoints/revision3/rungC_s0/ckpt_final.pt --name rungC_s0 --taus 0.5 0

Rung-B checkpoints need nothing special (their inputs are causal) and use agent_eval directly.
The patch is installed at import so the runner's spawned workers get it too.
"""

from __future__ import annotations

import experiments.run_sim as RS
from privileged.oracle_reward import wrap_builder

if not getattr(RS.build_risk_field, "__privileged__", False):
    RS.build_risk_field = wrap_builder(RS.build_risk_field)


def main(argv=None) -> int:
    from experiments.revision1.agent_eval import main as agent_eval_main

    return agent_eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
