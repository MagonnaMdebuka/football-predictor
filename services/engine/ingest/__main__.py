"""Allow running ingest as a module: python -m services.engine.ingest."""

from services.engine.ingest.cli import ingest

if __name__ == "__main__":
    ingest()
