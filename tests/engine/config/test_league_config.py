"""Tests for league configuration and capability flags."""

from services.engine.config.league_defaults import LeagueConfig, get_league_config


class TestLeagueConfigFlags:
    """Capability and ship flags behave correctly."""

    def test_e0_has_corners_and_cards(self):
        cfg = get_league_config("E0")
        assert cfg.has_corners is True
        assert cfg.has_cards is True

    def test_e0_does_not_ship_corners_or_cards(self):
        cfg = get_league_config("E0")
        assert cfg.ship_corners is False
        assert cfg.ship_cards is False

    def test_default_ship_flags_are_false(self):
        cfg = LeagueConfig(league_code="X0", league_name="Test", xi=0.005)
        assert cfg.ship_corners is False
        assert cfg.ship_cards is False
        assert cfg.has_corners is False
        assert cfg.has_cards is False

    def test_ship_can_be_enabled(self):
        cfg = LeagueConfig(
            league_code="X0", league_name="Test", xi=0.005,
            has_corners=True, ship_corners=True,
        )
        assert cfg.ship_corners is True
        assert cfg.has_corners is True

    def test_has_without_ship(self):
        """has_corners=True with ship_corners=False means backtest runs but UI suppresses."""
        cfg = LeagueConfig(
            league_code="X0", league_name="Test", xi=0.005,
            has_corners=True, ship_corners=False,
        )
        assert cfg.has_corners is True
        assert cfg.ship_corners is False
