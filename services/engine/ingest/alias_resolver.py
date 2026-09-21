"""Team alias resolution: exact → normalised → fuzzy matching.

Resolution order:
1. Exact match on (source, raw_name) in team_aliases table
2. Normalised match (lowercase, accent-stripped, FC removed)
3. rapidfuzz token_sort_ratio: auto-accept if score >= threshold AND gap to next best >= min_gap
4. Return None for ambiguous/below-threshold matches (require manual review)
"""

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

from services.engine.ingest.normalise import normalise_team_name

logger = logging.getLogger(__name__)


@dataclass
class AliasMatch:
    team_id: int
    canonical_name: str
    score: float
    confirmed: bool


class AliasResolver:
    """In-memory alias resolver backed by the team_aliases table.

    Loads all aliases and team names once at construction, then resolves
    incoming raw names through a three-stage pipeline.
    """

    def __init__(
        self,
        teams: dict[int, str],
        aliases: dict[tuple[str, str], int | None],
        auto_accept_threshold: float = 92.0,
        min_gap: float = 3.0,
    ) -> None:
        """
        Args:
            teams: {team_id: canonical_name}
            aliases: {(source, raw_name): team_id_or_None}
            auto_accept_threshold: minimum fuzzy score for auto-acceptance
            min_gap: minimum gap to second-best score for auto-acceptance
        """
        self.teams = teams
        self.aliases = dict(aliases)
        self.auto_accept_threshold = auto_accept_threshold
        self.min_gap = min_gap

        # Build normalised lookup: normalised_name → team_id
        self._normalised_lookup: dict[str, int] = {}
        for team_id, canonical_name in teams.items():
            norm = normalise_team_name(canonical_name)
            self._normalised_lookup[norm] = team_id

        # Track new aliases discovered during resolution
        self.new_aliases: list[tuple[str, str, int, float, bool]] = []

    def resolve(self, source: str, raw_name: str) -> int | None:
        """Resolve a raw team name from a given source to a team_id.

        Returns the team_id if resolved, None if ambiguous or below threshold.
        """
        # Stage 1: exact match in alias table
        key = (source, raw_name)
        if key in self.aliases:
            return self.aliases[key]

        # Stage 2: normalised match against canonical names
        norm = normalise_team_name(raw_name)
        if norm in self._normalised_lookup:
            team_id = self._normalised_lookup[norm]
            self.aliases[key] = team_id
            self.new_aliases.append((source, raw_name, team_id, 100.0, True))
            logger.info("Normalised match: '%s' → '%s' (id=%d)", raw_name, self.teams[team_id], team_id)
            return team_id

        # Stage 3: fuzzy match
        scores: list[tuple[int, str, float]] = []
        for team_id, canonical_name in self.teams.items():
            score = fuzz.token_sort_ratio(norm, normalise_team_name(canonical_name))
            scores.append((team_id, canonical_name, score))

        scores.sort(key=lambda x: x[2], reverse=True)

        if not scores:
            return None

        best_id, best_name, best_score = scores[0]
        second_score = scores[1][2] if len(scores) > 1 else 0.0
        gap = best_score - second_score

        if best_score >= self.auto_accept_threshold and gap >= self.min_gap:
            self.aliases[key] = best_id
            self.new_aliases.append((source, raw_name, best_id, best_score, True))
            logger.info(
                "Fuzzy match: '%s' → '%s' (score=%.1f, gap=%.1f)",
                raw_name, best_name, best_score, gap,
            )
            return best_id

        # Below threshold or ambiguous — record for manual review
        logger.warning(
            "Unresolved alias: '%s' (best='%s' score=%.1f, gap=%.1f)",
            raw_name, best_name, best_score, gap,
        )
        self.aliases[key] = None
        if best_score > 0:
            self.new_aliases.append((source, raw_name, best_id, best_score, False))
        return None

    @classmethod
    def from_session(cls, session, config=None) -> "AliasResolver":
        """Build an AliasResolver from the current database state."""
        from services.engine.ingest.config import IngestConfig

        if config is None:
            config = IngestConfig()

        from db.models import Team, TeamAlias

        teams = {t.id: t.canonical_name for t in session.query(Team).all()}
        aliases: dict[tuple[str, str], int | None] = {}
        for a in session.query(TeamAlias).all():
            aliases[(a.source, a.raw_name)] = a.team_id if a.confirmed else None

        return cls(
            teams=teams,
            aliases=aliases,
            auto_accept_threshold=config.fuzzy_auto_accept_threshold,
            min_gap=config.fuzzy_min_gap,
        )

    def flush_new_aliases(self, session) -> int:
        """Write newly discovered aliases back to the database. Returns count written."""
        from sqlalchemy.dialects.postgresql import insert

        from db.models import TeamAlias

        count = 0
        for source, raw_name, team_id, score, confirmed in self.new_aliases:
            stmt = (
                insert(TeamAlias)
                .values(
                    source=source,
                    raw_name=raw_name,
                    team_id=team_id if confirmed else None,
                    score=score,
                    confirmed=confirmed,
                )
                .on_conflict_do_update(
                    index_elements=["source", "raw_name"],
                    set_={
                        "team_id": team_id if confirmed else None,
                        "score": score,
                        "confirmed": confirmed,
                    },
                )
            )
            session.execute(stmt)
            count += 1

        self.new_aliases.clear()
        return count
