# Phase 6: Deferred Requirements

## DB Constraint: team_strengths temporal integrity

When team strength parameters are stored in the database, a constraint must ensure that `team_strengths` records reference a `model_run` whose training data cutoff is strictly before the match date. This prevents using parameters fitted on data that includes the match being predicted.

**Implementation approach:**
- Add a `fitted_before` timestamp column to `model_runs` recording the training data cutoff
- Add a CHECK constraint or application-level validation ensuring `model_runs.fitted_before < matches.kickoff_utc` for any prediction
- The backtest harness already enforces this at the application level; the DB constraint provides defence-in-depth
