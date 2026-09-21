"""Tests for Dixon-Coles parameter pack/unpack and DixonColesParams."""

import numpy as np
import pytest

from services.engine.models.params import (
    DixonColesParams,
    pack,
    unpack,
    vector_length,
)

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Spurs"]


@pytest.fixture
def sample_params() -> DixonColesParams:
    attack = np.array([0.2, -0.1, 0.3, -0.4])
    defence = np.array([0.1, -0.2, 0.0, 0.1])
    return DixonColesParams(
        teams=TEAMS,
        mu=0.25,
        attack=attack,
        defence=defence,
        gamma=0.2,
        rho=-0.1,
    )


class TestPackUnpackRoundTrip:
    def test_round_trip_preserves_values(self, sample_params: DixonColesParams):
        vec = pack(sample_params)
        restored = unpack(vec, TEAMS)
        assert restored.mu == pytest.approx(sample_params.mu)
        assert restored.gamma == pytest.approx(sample_params.gamma)
        assert restored.rho == pytest.approx(sample_params.rho)
        np.testing.assert_allclose(restored.attack, sample_params.attack)
        np.testing.assert_allclose(restored.defence, sample_params.defence)

    def test_round_trip_preserves_teams(self, sample_params: DixonColesParams):
        vec = pack(sample_params)
        restored = unpack(vec, TEAMS)
        assert restored.teams == TEAMS


class TestSumToZero:
    def test_attack_sums_to_zero(self, sample_params: DixonColesParams):
        assert sample_params.attack.sum() == pytest.approx(0.0)

    def test_defence_sums_to_zero(self, sample_params: DixonColesParams):
        assert sample_params.defence.sum() == pytest.approx(0.0)

    def test_unpack_enforces_sum_to_zero(self):
        """Unpack derives last team so attack/defence always sum to zero."""
        teams = ["A", "B", "C"]
        vec = np.array([0.3, 0.5, -0.2, 0.1, 0.3, 0.15, -0.05])
        params = unpack(vec, teams)
        assert params.attack.sum() == pytest.approx(0.0)
        assert params.defence.sum() == pytest.approx(0.0)


class TestVectorLength:
    def test_four_teams(self):
        assert vector_length(4) == 9

    def test_twenty_teams(self):
        assert vector_length(20) == 41

    def test_packed_vector_matches(self, sample_params: DixonColesParams):
        vec = pack(sample_params)
        assert len(vec) == vector_length(len(TEAMS))


class TestTeamLookup:
    def test_team_attack(self, sample_params: DixonColesParams):
        assert sample_params.team_attack("Arsenal") == pytest.approx(0.2)
        assert sample_params.team_attack("Spurs") == pytest.approx(-0.4)

    def test_team_defence(self, sample_params: DixonColesParams):
        assert sample_params.team_defence("Chelsea") == pytest.approx(-0.2)

    def test_unknown_team_raises(self, sample_params: DixonColesParams):
        with pytest.raises(ValueError):
            sample_params.team_attack("Man City")
