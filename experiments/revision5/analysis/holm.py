"""Holm correction across the three pre-registered p-values (ANALYSIS_SPEC). Writes analysis/holm.json.

    python -m experiments.revision5.analysis.holm
"""

from __future__ import annotations

import json

from experiments.revision5.analysis.common import OUT, holm, write


def main() -> int:
    h1 = json.loads((OUT / "h1.json").read_text())
    h2 = json.loads((OUT / "h2.json").read_text())
    h3 = json.loads((OUT / "h3.json").read_text())
    p = {"H1": h1["binomial_p_lower_vs_0.05"], "H2": h2["vs_adaptive_classical"]["p_two_sided"],
         "H3": h3["spearman"]["p_rho_le_0.6"]}
    adj = holm(p)
    payload = {"alpha": 0.05, "raw_p": p, "holm_adjusted_p": adj,
               "reject_at_0.05": {k: bool(v < 0.05) for k, v in adj.items()},
               "notes": {"H1": "exact one-sided binomial test of the one-family-wins-all-four share against 0.05",
                         "H2": "bootstrap two-sided p of the pooled relative cost difference (policy vs adaptive classical)",
                         "H3": "bootstrap p for Spearman rho <= 0.6 (the Holm-tested component; the other H3 criteria "
                               "are pass/fail)"}}
    print(write("holm.json", payload, {"inputs": ["h1.json", "h2.json", "h3.json"]}))
    print(payload["raw_p"], payload["holm_adjusted_p"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
