"""Smoke test: verify engine subpackages are importable."""


def test_engine_imports():
    import services.engine
    import services.engine.backtest
    import services.engine.ingest
    import services.engine.markets
    import services.engine.models

    assert services.engine is not None
    assert services.engine.ingest is not None
    assert services.engine.models is not None
    assert services.engine.markets is not None
    assert services.engine.backtest is not None
