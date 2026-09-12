"""Building geometry and NLOS link classification for the urban grid.

Why this exists
---------------
The rural corridor only becomes disconnected below ~3 veh/km/lane, which is an
implausibly empty road. Buildings give a far more believable route to the same
regime: in a city, two vehicles 150 m apart on perpendicular streets cannot see
each other at all, so the network fragments at densities where the road is
visibly busy.

Model
-----
A Manhattan grid of street centrelines at ``x = i*block`` and ``y = j*block``,
with a rectangular building occupying the interior of each block inset by half
a street width. A link is line-of-sight only if both endpoints lie on the same
street:

* same vertical street   -> ``|dx| <= street_half_width``
* same horizontal street -> ``|dy| <= street_half_width``

Anything else has a building between the endpoints. For those links the signal
has to travel around the corner, so the propagation distance is the Manhattan
distance rather than the Euclidean one, and a knife-edge diffraction loss is
charged at the corner:

    excess_dB = PL(d_manhattan) - PL(d_euclidean) + corner_loss_dB

Expressing it as an *excess over the LOS model* means the dual-slope path loss,
shadowing and fading all keep working unchanged; only an additive term is new.

This is a deliberately simple geometric model. It does not do multi-corner
diffraction, over-rooftop propagation, or waveguiding along a street canyon.
Those omissions all make it pessimistic about NLOS range, which is the
conservative direction for a paper claiming a sparse-connectivity contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BuildingGrid:
    """Rectangular city blocks between a Manhattan grid of streets."""

    block_length_m: float
    street_half_width_m: float
    rows: int
    cols: int

    def is_nlos(self, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """``[N, M]`` boolean: is the link from each ``a`` to each ``b`` blocked?

        ``pos_a`` is ``[N, 2]`` and ``pos_b`` is ``[M, 2]``.
        """
        ax, ay = pos_a[:, 0][:, None], pos_a[:, 1][:, None]
        bx, by = pos_b[:, 0][None, :], pos_b[:, 1][None, :]
        dx = np.abs(ax - bx)
        dy = np.abs(ay - by)
        same_vertical_street = dx <= self.street_half_width_m
        same_horizontal_street = dy <= self.street_half_width_m
        return ~(same_vertical_street | same_horizontal_street)

    @staticmethod
    def manhattan_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """``[N, M]`` L1 distance: the path length around the corner."""
        ax, ay = pos_a[:, 0][:, None], pos_a[:, 1][:, None]
        bx, by = pos_b[:, 0][None, :], pos_b[:, 1][None, :]
        return np.abs(ax - bx) + np.abs(ay - by)

    @staticmethod
    def euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)

    def describe(self) -> str:
        return (
            f"BuildingGrid {self.rows}x{self.cols} blocks of {self.block_length_m:g} m, "
            f"street half-width {self.street_half_width_m:g} m"
        )


def build_building_grid(trace_meta: dict, street_half_width_m: float) -> BuildingGrid | None:
    """Construct the grid from a trace's metadata, or None for non-grid scenarios."""
    if trace_meta.get("kind") != "grid":
        return None
    return BuildingGrid(
        block_length_m=float(trace_meta.get("block_length_m", 200.0)),
        street_half_width_m=float(street_half_width_m),
        rows=int(trace_meta.get("grid_rows", 5)),
        cols=int(trace_meta.get("grid_cols", 5)),
    )
