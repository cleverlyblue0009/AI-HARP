"""Train an R8 diagnostic-ladder rung (privileged reward, optionally privileged input).

::

    D:/aiharp-gpu/Scripts/python.exe -m privileged.train_privileged --rung B --seed 0 --out checkpoints/revision3/rungB_s0
    D:/aiharp-gpu/Scripts/python.exe -m privileged.train_privileged --rung C --resume checkpoints/revision3/rungC_s1/ckpt_latest.pt

Identical to ``agents.train`` (same config, curriculum, budget, deterministic kernels)
except:

* rung B, C: the coverage term and tree credit use ORACLE relevance
  (privileged.oracle_reward.privileged_episode_rewards), and the per-cell training
  targets are 0.95 x best of flooding/DV-CAST in ORACLE RWCR on the training seeds
  (results/revision3/R8/oracle_coverage_targets.json);
* rung C: oracle relevance is appended as node feature ``extra_relevance_oracle``.

The checkpoint config records ``privileged: {rung, reward, targets, ORACLE_INPUT}``.
Patches are installed at import, so the rollout workers (which re-import this module
under spawn) are patched identically; the rung travels in AIHARP_PRIVILEGED_RUNG.
"""

from __future__ import annotations

import os
import sys

from common.config import PROJECT_ROOT

ENV = "AIHARP_PRIVILEGED_RUNG"
TARGETS = "results/revision3/R8/oracle_coverage_targets.json"


def _rung_from_argv() -> str | None:
    if "--rung" in sys.argv:
        return sys.argv[sys.argv.index("--rung") + 1].upper()
    return None


def install(rung: str) -> None:
    import agents.constrained_reward as cr
    import agents.train as T
    from privileged.oracle_reward import privileged_episode_rewards, wrap_builder

    if rung not in ("B", "C"):
        raise SystemExit(f"unknown rung {rung!r} (expected B or C)")
    cr.episode_rewards = privileged_episode_rewards
    if rung == "C" and not getattr(T.build_risk_field, "__privileged__", False):
        T.build_risk_field = wrap_builder(T.build_risk_field)


RUNG = os.environ.get(ENV) or _rung_from_argv()
if RUNG:
    os.environ[ENV] = RUNG          # inherited by the spawned rollout workers
    install(RUNG)


def main(argv: list[str] | None = None) -> int:
    import agents.train as T
    from privileged.oracle_reward import ORACLE_FEATURE

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--rung" not in argv:
        raise SystemExit("--rung B|C is required")
    i = argv.index("--rung")
    rung = argv[i + 1].upper()
    del argv[i:i + 2]
    if not (PROJECT_ROOT / TARGETS).exists():
        raise SystemExit(f"{TARGETS} missing: run python -m experiments.revision3.oracle_targets first")
    orig = T.load_yaml

    def load_yaml(name, *a, **k):
        cfg = orig(name, *a, **k)
        if name == "agent.yaml":
            cfg["objective"]["targets_path"] = TARGETS
            cfg["privileged"] = {"rung": rung, "reward": "oracle-relevance coverage and credit",
                                 "targets": TARGETS, "ORACLE_INPUT": rung == "C",
                                 "deployable": False if rung == "C" else "reward only (inputs causal)"}
            if rung == "C":
                cfg["graph"]["extra_node_features"] = [ORACLE_FEATURE]
        return cfg

    T.load_yaml = load_yaml          # a --resume run keeps its checkpoint's config instead
    return T.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
