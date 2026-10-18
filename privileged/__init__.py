"""TRAINING-ONLY privileged signals for the revision-3 R8 diagnostic ladder.

This package is the one sanctioned exemption to the oracle/causal boundary
(README "The oracle/causal split"): it imports ``hazard.oracle`` to build

* rung B -- the constrained objective's coverage term and tree credit computed from
  ORACLE relevance instead of causal relevance (policy inputs unchanged), and
* rung C -- additionally, oracle relevance as an extra node feature
  (``graph.extra_node_features = ["extra_relevance_oracle"]``). Rung C is a
  NON-DEPLOYABLE upper bound: its checkpoints carry ``privileged.ORACLE_INPUT: true``.

The exemption is one-way and enforced by tests/test_oracle_isolation.py: decision-time
packages (agents/, sim/, mobility/) may not import ``privileged``. Privileged behaviour
is installed only by this package's own entry points (``privileged.train_privileged``,
``privileged.eval_privileged``), which patch the training/evaluation pipeline at import
time -- so spawned rollout workers, which re-import the entry module, get the same
patches. Nothing here is used by any deployable result.
"""
