# Revision 9 — reading notes (step 0)

Branch `revision-9`, from `origin/revision-8` at b67f18c (working tree identical to it). Written before the addendum.
Revision 9 is **motivated by a post-hoc observation** (revision-8 `REVISION8_STATUS.md` §6, read after the registered H5
result): in Veins rural d=20 STRUCT-rand reached mean RWCR 0.897–0.933 > q* 0.879 at cost 0.28–0.46 but its actionable
deadline-miss rate was 0.54–0.85 > m* 0.404 (`results/revision8/analysis/h5.json` `per_run`). Revision-8 results stand
unchanged.

## 1. Actionable deadline-miss: exact definition (`analysis/metrics.py`)

* `actionable_deadline_miss_rate` (:246-307). Origination step s; ETA of every vehicle at s from `res.risk.evaluate`
  (:275-279); actionable = at-risk & (absent at s, or ETA at s ≥ reaction time) (:284-286). A vehicle is OK iff
  informed in time (:294), latency ≤ `hazard.safety_deadline_s` (:295) and, with `require_actionable_eta`, ETA at the
  informing step ≥ reaction time (:296-301, `_eta_at` :393-400). Miss = 1 − OK / actionable (:304).
* Context (`_build_context`, :107-140): at-risk set = `oracle.at_risk_set(rel)` (:110-113); informed in time = informed
  and **oracle** relevance > 0 at the informing step (:118-124); latency = informed step × dt − onset, clipped at 0
  (:126-128).
* Deadline settings (`configs/hazard.yaml:121-132`): severity bands (≥ 0.9: 4 s, ≥ 0.7: 6 s, ≥ 0.5: 10 s, else 15 s),
  `require_actionable_eta: true`, `reaction_time_s: 2.5`. `hazard.safety_deadline_s` comes from them
  (`hazard/model.py:225`).
* **Oracle-only inputs: exactly two** — the at-risk set and the "informed in time" relevance test. ETA at origination and
  at informing already come from the **causal** risk field (`res.risk` is the causal `RiskField`), and latency,
  deadlines and the reaction time are causal by construction.
* **Causal version: expressible exactly.** `agents/causal_miss.py` implements the same formula on a given relevance
  matrix and at-risk set; with the causal matrix (`RiskField.relevance_matrix`, `hazard/risk_field.py:240`) and causal
  at-risk set (`RiskField.at_risk_set`, :251-259) it is the causal actionable miss. `tests/test_causal_miss.py`: with the
  oracle matrix and set injected it equals the metric bit for bit (12 cases: four headline cells × flooding,
  slotted_1p 50, p_persistence 0.1 × seeds 100–101). `tests/test_oracle_isolation.py` checks it transitively.
* First sign (one episode, rural d=20, seed 100, RAR warm start): causal miss 0.126 vs oracle 0.344. The causal at-risk
  set and in-time test differ from the oracle's (revision-1 "risk estimation difficulty"); gate C quantifies this.

## 2. Revision-8 training fitness (`agents/train_structured.py` at b67f18c)

* `fitness` (:164-183): per group (scenario|density) pooled causal shortfall (Σtarget − Σcoverage)/Σtarget; feasible iff
  all ≤ 0 (:180); f = cost (Σtx/Σcausal at-risk) if feasible else 1e6 + Σ positive shortfall (:182). **No deadline term.**
* Training groups: the four headline (scenario, density) pairs, clear, fog_bank (:36), 3 pool seeds per group per
  generation (:38), 80 generations (:39). Coverage targets: 0.95 × the 32-seed ceilings of flooding/DV-CAST defaults
  (`results/coverage_targets.json`, `experiments/coverage_targets.py`).
* **Interpretation (not a conflict):** the instruction says "per training group (same groups as coverage_targets.json)";
  that file holds 96 groups, of which revision-8 training uses four. b_g is computed for those four, which are the
  groups the fitness evaluates.

## 3. CI workflows and gates (revision 8)

* `.github/workflows/r8-train.yml`, `r8-eval.yml`, `r8-veins.yml`, `r8-cd.yml`; request files `ci/r8-*.json`; outputs on
  branch `r8-ci` (the session's integration cannot dispatch workflows, so pushes of request files trigger them).
* Gate check 1 (classical slotted_1p n=8, rural d=20, seed 200 → `707d6533…`): every H5 and C/D runner
  (`experiments/revision8/veins/ci_veins.py:91-100`, `main_cd`). **Gate check 2** (AI-HARP ref_s0 bias +1, every new
  option off → `220dd1fd…`, reference committed in `results/revision7/veins/raw/r7_gate/`): **only the C/D runners ran it**
  (`ci_veins.py` `main_cd`); the revision-8 H5 runners ran check 1 only. Revision 9 runs both on every Veins runner.
* Push loop: `git commit || true`, then `git push` with retries and `exit 1` after 10 failures (`r8-veins.yml:98-102`).
  It does not verify that the outputs are on the branch afterwards; revision 9 adds that check.
* Decision-server logs: excluded by `.gitignore` (`results/revision8/veins/ci/*/bridge_logs/`,
  `results/revision8/veins/ci_cd/*/bridge_logs/`), but `git checkout origin/r8-ci -- <dir>` stages them anyway
  (revision-8 020edeb). Revision 9 adds the revision-9 paths to `.gitignore` and removes them from the index after every
  checkout from `r9-ci`.
* AI-HARP checkpoints for gate check 2: committed as `results/revision1/checkpoints/ref_s{0..4}_ckpt_final.pt` (hashes =
  `results/revision5/checkpoint_shas.json`; revision-8 D15).

## 4. Veins seeds

* Tier A 200–209: classical (revision 5), rules and AI-HARP (revision 6/7), C/D (revision 8). No STRUCT replay.
* Tier B 210–219: STRUCT plain/rand s0–4 (revision-8 H5, D12).
* **Tier B 220–229: no STRUCT replay of any kind.** Classical revision-5 audit replays exist (`results/revision5/veins/
  veins_rows.jsonl`, 6,600 rows = 55 × 4 cells × 30 seeds), as for 210–219.
* Frozen Veins thresholds (`results/revision8/veins_thresholds.json`, from Tier A classical): rural d=20 q* 0.879,
  m* 0.404; urban NLOS d=20 q* 0.498, m* 0.602.

## 5. Nothing declared missing

Every file this revision needs was found in the repository (checkpoints, warm start, thresholds, training pool traces are
regenerated deterministically on runners as in revision 8). Nothing has been searched for and not found.
