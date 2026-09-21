"""Text normalisation, timezone conversion, and season helpers."""

import re
import unicodedata
from datetime import date, datetime, time, timezone

from zoneinfo import ZoneInfo

UK_TZ = ZoneInfo("Europe/London")


def normalise_team_name(name: str) -> str:
    """Normalise a team name for matching: lowercase, strip accents, remove FC/AFC suffixes."""
    # Strip leading/trailing whitespace
    name = name.strip()
    # NFKD decomposition to separate accents from base characters
    name = unicodedata.normalize("NFKD", name)
    # Remove combining characters (accents)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Lowercase
    name = name.lower()
    # Remove common suffixes
    name = re.sub(r"\b(fc|afc|cf|sc)\b", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_kickoff_utc(match_date: date, time_str: str | None) -> tuple[datetime, bool]:
    """Convert a match date and optional time (UK local) to UTC datetime.

    Returns (datetime_utc, kickoff_time_known).
    If time_str is None or empty, defaults to 15:00 UK time with kickoff_time_known=False.
    """
    if time_str and time_str.strip():
        parts = time_str.strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        kickoff_time_known = True
    else:
        hour, minute = 15, 0
        kickoff_time_known = False

    local_dt = datetime.combine(match_date, time(hour, minute), tzinfo=UK_TZ)
    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt, kickoff_time_known


def parse_csv_date(date_str: str) -> date:
    """Parse a date string from football-data.co.uk CSV.

    Handles both dd/mm/yy and dd/mm/yyyy formats.
    """
    date_str = date_str.strip()
    parts = date_str.split("/")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse date: {date_str}")

    day = int(parts[0])
    month = int(parts[1])
    year = int(parts[2])

    # Two-digit year: 00-99 → 2000-2099 (football data range)
    if year < 100:
        year += 2000

    return date(year, month, day)


def season_label_from_date(match_date: date) -> str:
    """Derive a season label (e.g., '2024-25') from a match date.

    Season boundary: August 1. Matches before August belong to the previous season.
    """
    if match_date.month >= 8:
        start_year = match_date.year
    else:
        start_year = match_date.year - 1
    end_year = start_year + 1
    return f"{start_year}-{end_year % 100:02d}"


def is_no_crowd(match_date: date) -> bool:
    """Check whether a match date falls in the COVID no-crowd period."""
    from services.engine.ingest.constants import NO_CROWD_END, NO_CROWD_START

    return NO_CROWD_START <= match_date <= NO_CROWD_END
