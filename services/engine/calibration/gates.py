"""Publication gate checks for market quality."""

from __future__ import annotations

PUBLICATION_GATES = {
    "min_settled": 500,
    "max_decile_error_pp": 5.0,
    "min_bin_count": 30,
    "must_beat_baseline": True,
    "requires_direct_data": True,
}


def check_publication_gates(
    reliability_data: dict,
    n_settled: int,
    beats_baseline: bool,
    has_direct_data: bool,
    min_bin_count: int | None = None,
) -> list[dict]:
    """Check all 4 publication gates.

    Parameters
    ----------
    reliability_data : output of compute_reliability_data()
    n_settled : number of settled predictions
    beats_baseline : whether model outperforms baseline
    has_direct_data : whether calibration comes from direct (not inferred) data
    min_bin_count : minimum observations per bin to include in the decile
        error gate. Bins below this threshold are skipped. Defaults to
        PUBLICATION_GATES["min_bin_count"] (30).

    Returns
    -------
    List of {name, passed, message} dicts for each gate.
    """
    if min_bin_count is None:
        min_bin_count = PUBLICATION_GATES["min_bin_count"]

    results = []

    # Gate 1: minimum settled predictions
    min_settled = PUBLICATION_GATES["min_settled"]
    results.append({
        "name": "min_settled",
        "passed": n_settled >= min_settled,
        "message": f"{n_settled} / {min_settled} settled predictions",
    })

    # Gate 2: max decile calibration error (only bins with >= min_bin_count)
    max_error_pp = PUBLICATION_GATES["max_decile_error_pp"]
    bins = reliability_data.get("bins", [])
    eligible = [
        b for b in bins
        if b.get("count", 0) >= min_bin_count
    ]
    if eligible:
        cal_error = max(
            abs(b["predicted_mean"] - b["observed_mean"]) for b in eligible
        )
        cal_error_pp = cal_error * 100
        n_skipped = len(bins) - len(eligible)
        msg = f"Max decile error {cal_error_pp:.1f}pp (limit {max_error_pp}pp)"
        if n_skipped > 0:
            msg += f" [{n_skipped} sparse bins skipped, min {min_bin_count}]"
    else:
        # Fall back to the pre-computed calibration_error if no bins provided
        cal_error_pp = reliability_data["calibration_error"] * 100
        msg = f"Max decile error {cal_error_pp:.1f}pp (limit {max_error_pp}pp)"

    results.append({
        "name": "max_decile_error",
        "passed": cal_error_pp <= max_error_pp,
        "message": msg,
    })

    # Gate 3: must beat baseline
    results.append({
        "name": "beats_baseline",
        "passed": beats_baseline,
        "message": "Model beats baseline" if beats_baseline else "Model does not beat baseline",
    })

    # Gate 4: requires direct data
    results.append({
        "name": "direct_data",
        "passed": has_direct_data,
        "message": (
            "Calibrated from direct data"
            if has_direct_data
            else "Calibrated from inferred data only"
        ),
    })

    return results
