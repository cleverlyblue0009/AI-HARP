"""The oracle must never reach a decision-time input.

``hazard/oracle.py`` labels vehicles using their *realised* trajectories. That
information does not exist when a policy decides. If it leaked into a node
feature, a reward term or any policy input, the controller would be reading the
future and every result in the paper would be invalid -- silently, because the
numbers would simply look good.

These tests enforce the boundary structurally rather than by convention. They
parse the actual import graph, so they cannot be satisfied by a comment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Packages that run at decision time. Nothing here may see ground truth.
DECISION_TIME_PACKAGES = ("agents", "sim", "mobility")

#: The forbidden module, and the names it exports.
ORACLE_MODULE = "hazard.oracle"
ORACLE_NAMES = {
    "OracleRiskField", "build_oracle_risk_field", "oracle_horizon_s",
    "relevance_oracle", "estimation_agreement", "encounter_time_s",
}


def _python_files(package: str) -> list[Path]:
    return sorted((ROOT / package).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by a file, from its AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


@pytest.mark.parametrize("package", DECISION_TIME_PACKAGES)
def test_decision_time_code_never_imports_the_oracle(package):
    offenders = []
    for path in _python_files(package):
        imported = _imported_modules(path)
        if any(m == ORACLE_MODULE or m.startswith(ORACLE_MODULE + ".") for m in imported):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"{package}/ imports {ORACLE_MODULE}: {offenders}. "
        "Oracle relevance is ground truth computed from realised trajectories; "
        "it must never reach a decision-time input. Use hazard.risk_field "
        "(causal) instead."
    )


@pytest.mark.parametrize("package", DECISION_TIME_PACKAGES)
def test_decision_time_code_never_references_oracle_names(package):
    """Catch a re-export or an alias that dodges the module-name check."""
    offenders = []
    for path in _python_files(package):
        src = path.read_text(encoding="utf-8")
        for name in ORACLE_NAMES:
            if name in src:
                offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert not offenders, f"oracle symbols referenced in {package}/: {offenders}"


def test_hazard_package_does_not_re_export_the_oracle():
    """``hazard/__init__`` must stay clean.

    Re-exporting would make ``from hazard import ...`` a back door: sim/ and
    agents/ import hazard.model legitimately, and a package-level re-export
    would put the oracle one attribute lookup away.
    """
    src = (ROOT / "hazard" / "__init__.py").read_text(encoding="utf-8")
    assert "oracle" not in src.lower(), (
        "hazard/__init__.py must not import or re-export the oracle; "
        "evaluation code should import hazard.oracle explicitly."
    )


def test_only_analysis_uses_the_oracle():
    """Positive check: the oracle is actually used where it should be."""
    users = [
        str(p.relative_to(ROOT))
        for p in _python_files("analysis")
        if any(m.startswith(ORACLE_MODULE) for m in _imported_modules(p))
    ]
    assert users, (
        "Nothing under analysis/ imports the oracle -- the at-risk set is "
        "presumably still being computed causally, which is the bug this split exists to fix."
    )


def test_engine_result_carries_no_oracle_state():
    """The engine's output must not smuggle ground truth to the policy layer."""
    import sim.engine as engine

    src = Path(engine.__file__).read_text(encoding="utf-8")
    for token in ("oracle", "encounter_time", "realised", "ground_truth"):
        assert token not in src.lower(), f"sim/engine.py references {token!r}"


# ---------------------------------------------------------------------------
# Revision-3 R8: the privileged/ package is the ONE sanctioned oracle user on the
# training side (oracle-relevance reward, and the non-deployable rung-C oracle input).
# The exemption is one-way: decision-time code may not import it, and nothing outside
# privileged/ (and these tests) may import it either.
# ---------------------------------------------------------------------------
PRIVILEGED = "privileged"


@pytest.mark.parametrize("package", DECISION_TIME_PACKAGES)
def test_decision_time_code_never_imports_privileged(package):
    offenders = [str(p.relative_to(ROOT)) for p in _python_files(package)
                 if any(m == PRIVILEGED or m.startswith(PRIVILEGED + ".") for m in _imported_modules(p))]
    assert not offenders, (f"{package}/ imports {PRIVILEGED}: {offenders}. The privileged "
                           "package exists only for the R8 diagnostic ladder's own entry points.")


@pytest.mark.parametrize("package", DECISION_TIME_PACKAGES)
def test_decision_time_code_never_names_the_privileged_feature(package):
    offenders = [str(p.relative_to(ROOT)) for p in _python_files(package)
                 if "extra_relevance_oracle" in p.read_text(encoding="utf-8")]
    assert not offenders, f"privileged feature named in {package}/: {offenders}"


def test_only_privileged_entry_points_import_privileged():
    import subprocess

    allowed = {PRIVILEGED, "tests"}
    offenders = []
    try:  # the repository's own Python files (tracked or new), not caches or build dirs
        listed = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                                          "*.py"], cwd=ROOT, text=True).split()
        files = [ROOT / f for f in listed]
    except Exception:  # pragma: no cover
        files = [p for p in ROOT.rglob("*.py") if "cache" not in p.parts]
    for path in files:
        rel = path.relative_to(ROOT)
        if rel.parts[0] in allowed or not path.exists():
            continue
        if any(m == PRIVILEGED or m.startswith(PRIVILEGED + ".") for m in _imported_modules(path)):
            offenders.append(str(rel))
    assert not offenders, f"only privileged/ may import privileged; found: {offenders}"


def test_privileged_package_is_where_the_training_side_oracle_lives():
    """Positive check: the exemption is used, and documented in the package."""
    users = [p for p in _python_files(PRIVILEGED)
             if any(m.startswith(ORACLE_MODULE) for m in _imported_modules(p))]
    assert users, "privileged/ no longer imports the oracle; drop the exemption"
    doc = (ROOT / PRIVILEGED / "__init__.py").read_text(encoding="utf-8")
    assert "exemption" in doc.lower() and "NON-DEPLOYABLE" in doc
