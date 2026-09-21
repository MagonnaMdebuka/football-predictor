"""Constants for football data ingestion.

CSV column mappings, season codes, and no-crowd date ranges.
"""

from datetime import date

# football-data.co.uk CSV column mapping → our DB columns
CSV_COLUMN_MAP: dict[str, str] = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "ft_home_goals",
    "FTAG": "ft_away_goals",
    "FTR": "ft_result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# Season codes used in football-data.co.uk URLs: {season_label: url_code}
# URL pattern: https://www.football-data.co.uk/mmz4281/{code}/E0.csv
SEASON_CODES: dict[str, str] = {
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
    "2026-27": "2627",
}

# Season start/end dates (approximate, for seeding)
SEASON_DATES: dict[str, tuple[str, str]] = {
    "2019-20": ("2019-08-09", "2020-07-26"),
    "2020-21": ("2020-09-12", "2021-05-23"),
    "2021-22": ("2021-08-13", "2022-05-22"),
    "2022-23": ("2022-08-05", "2023-05-28"),
    "2023-24": ("2023-08-11", "2024-05-19"),
    "2024-25": ("2024-08-16", "2025-05-25"),
    "2025-26": ("2025-08-16", "2026-05-24"),
    "2026-27": ("2026-08-15", "2027-05-23"),
}

# COVID no-crowd period: matches in this range get no_crowd=True
NO_CROWD_START = date(2020, 6, 17)
NO_CROWD_END = date(2021, 5, 17)  # Partial crowds returned from 17 May 2021

# Premier League identifiers
PL_FD_COUK_CODE = "E0"
PL_FD_ORG_CODE = 2021  # football-data.org competition ID

# football-data.org status values
FD_ORG_STATUS_FINISHED = "FINISHED"
FD_ORG_STATUS_SCHEDULED = "SCHEDULED"
FD_ORG_STATUS_TIMED = "TIMED"
FD_ORG_STATUS_IN_PLAY = "IN_PLAY"
FD_ORG_STATUS_PAUSED = "PAUSED"

# Minimum columns required in a valid CSV row
REQUIRED_CSV_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
