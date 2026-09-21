"""Parse football-data.co.uk CSV files into normalised row dicts.

Handles BOM (utf-8-sig), latin-1 fallback, blank rows, dd/mm/yy and dd/mm/yyyy
date formats, missing Time column, and column mapping to our schema.
"""

import csv
import hashlib
import io
import logging
from pathlib import Path
from typing import Any

from services.engine.ingest.constants import CSV_COLUMN_MAP, REQUIRED_CSV_COLUMNS
from services.engine.ingest.normalise import is_no_crowd, parse_csv_date, parse_kickoff_utc

logger = logging.getLogger(__name__)


def _read_csv_text(file_path: Path) -> str:
    """Read a CSV file, trying utf-8-sig first then latin-1."""
    try:
        return file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        logger.info("UTF-8 decode failed for %s, falling back to latin-1", file_path)
        return file_path.read_text(encoding="latin-1")


def compute_checksum(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv(file_path: Path) -> list[dict[str, Any]]:
    """Parse a football-data.co.uk CSV file into a list of normalised row dicts.

    Each returned dict contains:
    - date (datetime.date)
    - time (str or None)
    - home_team (str)
    - away_team (str)
    - ft_home_goals (int)
    - ft_away_goals (int)
    - kickoff_utc (datetime, UTC)
    - kickoff_time_known (bool)
    - no_crowd (bool)
    - stat columns (int or None): ht goals, shots, corners, fouls, yellows, reds
    - referee (str or None)
    - raw (dict): original row as parsed from CSV
    """
    text = _read_csv_text(file_path)
    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None:
        logger.error("No headers found in %s", file_path)
        return []

    # Validate required columns
    headers = set(reader.fieldnames)
    missing = REQUIRED_CSV_COLUMNS - headers
    if missing:
        logger.error("Missing required columns in %s: %s", file_path, missing)
        return []

    rows: list[dict[str, Any]] = []
    for line_num, raw_row in enumerate(reader, start=2):
        # Skip blank rows
        if not raw_row.get("Date") or not raw_row.get("HomeTeam"):
            continue

        try:
            row = _parse_row(raw_row)
            row["raw"] = {k: v for k, v in raw_row.items() if k and v}
            rows.append(row)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping line %d in %s: %s", line_num, file_path, exc)

    logger.info("Parsed %d rows from %s", len(rows), file_path.name)
    return rows


def _parse_row(raw: dict[str, str]) -> dict[str, Any]:
    """Parse a single CSV row into our normalised format."""
    match_date = parse_csv_date(raw["Date"])
    time_str = raw.get("Time", "").strip() or None

    kickoff_utc, kickoff_time_known = parse_kickoff_utc(match_date, time_str)

    row: dict[str, Any] = {
        "date": match_date,
        "time": time_str,
        "home_team": raw["HomeTeam"].strip(),
        "away_team": raw["AwayTeam"].strip(),
        "ft_home_goals": _safe_int(raw["FTHG"]),
        "ft_away_goals": _safe_int(raw["FTAG"]),
        "kickoff_utc": kickoff_utc,
        "kickoff_time_known": kickoff_time_known,
        "no_crowd": is_no_crowd(match_date),
    }

    # Optional stat columns
    for csv_col, db_col in CSV_COLUMN_MAP.items():
        if csv_col in ("Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTR"):
            continue
        if csv_col == "Referee":
            row["referee"] = raw.get(csv_col, "").strip() or None
        else:
            row[db_col] = _safe_int(raw.get(csv_col))

    return row


def _safe_int(value: str | None) -> int | None:
    """Convert a string to int, returning None for empty/invalid values."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None
