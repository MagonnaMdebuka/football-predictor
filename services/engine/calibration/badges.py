"""Quality badge assignment for markets."""

from __future__ import annotations


def compute_quality_badge(
    n_seasons: int,
    has_direct_data: bool,
    is_cross_competition: bool = False,
) -> str:
    """Return 'green', 'amber', or 'grey' badge.

    green: 3+ seasons of direct data
    amber: 1-2 seasons or cross-competition inference
    grey: insufficient data
    """
    if not has_direct_data:
        return "grey"

    if is_cross_competition:
        return "amber"

    if n_seasons >= 3:
        return "green"
    elif n_seasons >= 1:
        return "amber"
    else:
        return "grey"
