"""FCD warm-up handling: recorded from t = 0, or from the warm-up onward."""

from __future__ import annotations

import numpy as np

from mobility.fcd import parse_fcd


def _fcd(path, times):
    rows = ["<fcd-export>"]
    for t in times:
        rows.append(f'  <timestep time="{t:.2f}">')
        rows.append(f'    <vehicle id="v0" x="{10 * t:.2f}" y="0.00" angle="90.00" type="car" '
                    f'speed="10.00" lane="e0_0"/>')
        rows.append("  </timestep>")
    rows.append("</fcd-export>")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_export_from_zero_drops_the_warmup(tmp_path):
    times = [round(0.1 * k, 2) for k in range(0, 60)]          # 0.0 .. 5.9 s
    tr = parse_fcd(_fcd(tmp_path / "a.xml", times), scenario_name="t", dt=0.1, warmup_s=2.0)
    assert tr.x.shape[0] == 40                                  # 2.0 .. 5.9 s kept
    assert np.isclose(tr.x[0, 0], 20.0)


def test_export_beginning_at_the_warmup_keeps_every_step(tmp_path):
    times = [round(2.0 + 0.1 * k, 2) for k in range(0, 40)]    # recorded from 2.0 s
    tr = parse_fcd(_fcd(tmp_path / "b.xml", times), scenario_name="t", dt=0.1, warmup_s=2.0)
    assert tr.x.shape[0] == 40
    assert np.isclose(tr.x[0, 0], 20.0)
