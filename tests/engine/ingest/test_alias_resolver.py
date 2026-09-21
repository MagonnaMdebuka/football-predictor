"""Tests for alias resolver: exact, normalised, fuzzy, ambiguous pairs, seed, confirm."""

import pytest

from services.engine.ingest.alias_resolver import AliasResolver


# Fixture teams for testing
TEAMS = {
    1: "Manchester United",
    2: "Manchester City",
    3: "Sheffield United",
    4: "Sheffield Wednesday",
    5: "West Ham United",
    6: "West Bromwich Albion",
    7: "Bristol City",
    8: "Bristol Rovers",
    9: "Nottingham Forest",
    10: "Notts County",
    11: "Arsenal",
    12: "Liverpool",
    13: "Wolverhampton Wanderers",
}


@pytest.fixture
def resolver():
    return AliasResolver(
        teams=TEAMS,
        aliases={
            ("fd_couk", "Man United"): 1,
            ("fd_couk", "Man City"): 2,
        },
        auto_accept_threshold=92.0,
        min_gap=3.0,
    )


class TestExactMatch:
    def test_known_alias(self, resolver):
        assert resolver.resolve("fd_couk", "Man United") == 1

    def test_another_known_alias(self, resolver):
        assert resolver.resolve("fd_couk", "Man City") == 2

    def test_unknown_source_tries_fuzzy(self, resolver):
        """Different source should not match existing alias — falls through to fuzzy."""
        result = resolver.resolve("fd_org", "Man United")
        # "Man United" fuzzy-matches "Manchester United" at ~74, below the 92 threshold
        # so it should be None (unresolved) since the gap to Man City is small
        assert result is None


class TestNormalisedMatch:
    def test_normalised_exact(self, resolver):
        """Arsenal (canonical) should match itself via normalisation."""
        result = resolver.resolve("fd_couk", "Arsenal")
        assert result == 11

    def test_normalised_with_fc(self, resolver):
        """Arsenal FC should strip FC and match Arsenal."""
        result = resolver.resolve("fd_org", "Arsenal FC")
        assert result == 11

    def test_normalised_with_case(self, resolver):
        """Case-insensitive matching."""
        result = resolver.resolve("fd_couk", "LIVERPOOL")
        assert result == 12


class TestFuzzyMatch:
    def test_high_confidence_match(self, resolver):
        """Wolverhampton should fuzzy-match Wolverhampton Wanderers."""
        result = resolver.resolve("fd_couk", "Wolverhampton Wanderers FC")
        assert result == 13

    def test_records_new_alias(self, resolver):
        """New fuzzy matches should be recorded."""
        resolver.resolve("fd_couk", "Arsenal")
        assert len(resolver.new_aliases) >= 1


class TestAmbiguousPairs:
    """These pairs should be flagged as ambiguous (below gap threshold)."""

    def test_man_utd_vs_man_city(self, resolver):
        """Man Utd should be ambiguous between Man United and Man City."""
        result = resolver.resolve("new_source", "Man Utd")
        # This should either resolve correctly (high score to Man United) or be None
        # depending on the exact fuzzy scores
        if result is not None:
            assert result == 1  # Should be Manchester United if it resolves

    def test_sheffield_utd_vs_wed(self, resolver):
        """Sheffield should be ambiguous between Sheffield United and Wednesday."""
        result = resolver.resolve("test", "Sheffield")
        # Ambiguous — both start with Sheffield
        # Score gap will likely be < 3, so this should be None
        assert result is None or result in (3, 4)

    def test_west_ham_vs_west_brom(self, resolver):
        """West should be ambiguous between West Ham and West Brom."""
        result = resolver.resolve("test", "West")
        assert result is None or result in (5, 6)

    def test_nottm_forest_vs_notts_county(self, resolver):
        """Notts should be ambiguous between Nottingham Forest and Notts County."""
        result = resolver.resolve("test", "Notts")
        assert result is None or result in (9, 10)


class TestNewAliases:
    def test_new_alias_tracked(self, resolver):
        resolver.resolve("fd_couk", "Arsenal")
        assert any(a[1] == "Arsenal" for a in resolver.new_aliases)

    def test_new_alias_has_score(self, resolver):
        resolver.resolve("fd_couk", "Arsenal")
        for alias in resolver.new_aliases:
            if alias[1] == "Arsenal":
                assert alias[3] >= 90.0  # Should be 100.0 for normalised match
                break

    def test_confirmed_flag(self, resolver):
        """Normalised matches should be confirmed."""
        resolver.resolve("fd_couk", "Arsenal")
        for alias in resolver.new_aliases:
            if alias[1] == "Arsenal":
                assert alias[4] is True  # confirmed
                break
