# Archived results

## `runs_metrics_v1_pre_oracle.csv`

360 rows, `metrics_version = 1` (implicit -- the column did not exist yet).

**Do not plot these alongside current results.** They were computed before the
oracle/causal split, so their `rwcr`, `at_risk_coverage`, `tir_*` and
`deadline_miss_rate` use the *causal* at-risk set. On rural that reads about
0.044 RWCR higher than the current definition; on `urban_nlos` it reads 0.29
where the current definition reads 0.98.

Kept because they are the evidence behind the Phase 4 baseline table as
originally reported, and because deleting a result to make a story tidier is
exactly the wrong instinct. Regenerate with `experiments/compare.py` to get
`metrics_version = 2` rows.
