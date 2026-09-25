"""Tests for count model parameters: pack/unpack round-trip and sum-to-zero."""

import numpy as np
import pytest

from services.engine.models.count_params import (
    CountModelParams,
    pack,
    unpack,
    vector_length,
)


TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]
REFEREES = ["RefA", "RefB", "RefC"]


def _make_params(
    teams: list[str] | None = None,
    referees: list[str] | None = None,
    alpha: float = 0.5,
) -> CountModelParams:
    """Build a CountModelParams with known values."""
    teams = teams or TEAMS
    n = len(teams)
    attack = np.zeros(n, dtype=np.float64)
    attack[: n - 1] = [0.2, -0.1, 0.05][: n - 1]
    attack[n - 1] = -attack[: n - 1].sum()

    defence = np.zeros(n, dtype=np.float64)
    defence[: n - 1] = [-0.15, 0.1, 0.0][: n - 1]
    defence[n - 1] = -defence[: n - 1].sum()

    ref_list = referees or []
    r = len(ref_list)
    referee_effect = np.array([], dtype=np.float64)
    if r > 1:
        referee_effect = np.zeros(r, dtype=np.float64)
        referee_effect[: r - 1] = [0.1, -0.05][: r - 1]
        referee_effect[r - 1] = -referee_effect[: r - 1].sum()
    elif r == 1:
        referee_effect = np.zeros(1, dtype=np.float64)

    return CountModelParams(
        teams=teams,
        mu=0.2,
        attack=attack,
        defence=defence,
        gamma=0.15,
        alpha=alpha,
        referees=ref_list,
        referee_effect=referee_effect,
    )


class TestVectorLength:
    def test_no_referees(self):
        assert vector_length(4) == 10  # 2*4 + 2

    def test_with_referees(self):
        assert vector_length(4, 3) == 12  # 2*4 + 2 + 2

    def test_single_referee(self):
        assert vector_length(4, 1) == 10  # 2*4 + 2 + 0

    def test_two_referees(self):
        assert vector_length(4, 2) == 11  # 2*4 + 2 + 1


class TestPackUnpack:
    def test_round_trip_no_referees(self):
        params = _make_params()
        vec = pack(params)
        restored = unpack(vec, params.teams)

        assert restored.mu == pytest.approx(params.mu)
        assert restored.gamma == pytest.approx(params.gamma)
        assert restored.alpha == pytest.approx(params.alpha)
        np.testing.assert_allclose(restored.attack, params.attack)
        np.testing.assert_allclose(restored.defence, params.defence)
        assert restored.referees == []

    def test_round_trip_with_referees(self):
        params = _make_params(referees=REFEREES)
        vec = pack(params)
        restored = unpack(vec, params.teams, params.referees)

        assert restored.mu == pytest.approx(params.mu)
        assert restored.gamma == pytest.approx(params.gamma)
        assert restored.alpha == pytest.approx(params.alpha)
        np.testing.assert_allclose(restored.attack, params.attack)
        np.testing.assert_allclose(restored.defence, params.defence)
        np.testing.assert_allclose(restored.referee_effect, params.referee_effect)

    def test_vector_length_matches_pack(self):
        for r in [0, 1, 3]:
            refs = [f"Ref{i}" for i in range(r)] if r else None
            params = _make_params(referees=refs)
            vec = pack(params)
            assert len(vec) == vector_length(len(params.teams), len(params.referees))

    def test_sum_to_zero_attack(self):
        params = _make_params()
        assert params.attack.sum() == pytest.approx(0.0, abs=1e-12)

        vec = pack(params)
        restored = unpack(vec, params.teams)
        assert restored.attack.sum() == pytest.approx(0.0, abs=1e-12)

    def test_sum_to_zero_defence(self):
        params = _make_params()
        assert params.defence.sum() == pytest.approx(0.0, abs=1e-12)

        vec = pack(params)
        restored = unpack(vec, params.teams)
        assert restored.defence.sum() == pytest.approx(0.0, abs=1e-12)

    def test_sum_to_zero_referee_effect(self):
        params = _make_params(referees=REFEREES)
        assert params.referee_effect.sum() == pytest.approx(0.0, abs=1e-12)

        vec = pack(params)
        restored = unpack(vec, params.teams, params.referees)
        assert restored.referee_effect.sum() == pytest.approx(0.0, abs=1e-12)

    def test_single_referee_effect_is_zero(self):
        params = _make_params(referees=["OnlyRef"])
        vec = pack(params)
        restored = unpack(vec, params.teams, ["OnlyRef"])
        assert len(restored.referee_effect) == 1
        assert restored.referee_effect[0] == pytest.approx(0.0)


class TestTeamLookups:
    def test_team_attack(self):
        params = _make_params()
        for i, team in enumerate(params.teams):
            assert params.team_attack(team) == pytest.approx(float(params.attack[i]))

    def test_team_defence(self):
        params = _make_params()
        for i, team in enumerate(params.teams):
            assert params.team_defence(team) == pytest.approx(float(params.defence[i]))

    def test_ref_effect_known(self):
        params = _make_params(referees=REFEREES)
        for i, ref in enumerate(params.referees):
            assert params.ref_effect(ref) == pytest.approx(
                float(params.referee_effect[i])
            )

    def test_ref_effect_unknown_returns_zero(self):
        params = _make_params(referees=REFEREES)
        assert params.ref_effect("UnknownRef") == 0.0
