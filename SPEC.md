# Build: football match prediction platform

## Your role
Senior engineer shipping a statistical modelling system. Correctness of the maths
and honesty of the output matter more than feature breadth. Where a decision is
genuinely ambiguous, pick the option you would defend and record it in DECISIONS.md.
Do not ask me to choose things you can decide yourself.

## What to build
A Forebet-style web app that predicts football match outcomes across roughly 20 top
competitions and publishes calibrated probabilities for every major betting market:
match result, over/under goals, both teams to score, correct score, Asian and European
handicaps, double chance, draw no bet, team totals, clean sheets, winning margin,
odd/even, corners, cards and bookings, half-time markets, half-time/full-time, and
first goal timing.

Core design rule: ONE model, ONE source of truth. A single Dixon-Coles score-probability
grid produces every goals-derived market, so no two numbers on a page can contradict
each other. Corners and cards get their own parallel count models.

This cannot and will not be accurate all the time. Build the app so it says so:
confidence bands on every probability, a public accuracy page scoring the model on its
own settled predictions, probabilistic language only, never advisory language, no
affiliate links, and a responsible gambling notice in the footer.

## Stack
Python 3.12 (numpy, scipy, pandas, statsmodels) for all modelling. FastAPI for the read
API. PostgreSQL 16 as the single store. Redis for caching and rate-limit budgets.
Next.js 15 App Router with TypeScript, Tailwind and shadcn/ui for the web app. Recharts
for visuals. Alembic for migrations. Docker Compose for local dev.

Layout:
  apps/web/              Next.js frontend
  services/api/          FastAPI read API
  services/engine/       ingest/, models/, markets/, backtest/
  db/migrations/         Alembic
  docker-compose.yml, Makefile

## Data sources, all free
1. football-data.co.uk season CSVs. No key, no rate limit, back to 1993. This is the
   PRIMARY training set and the ONLY free source with per-match corners (HC, AC),
   cards (HY, AY, HR, AR), shots (HS, AS, HST, AST), fouls and referee name. Its main
   leagues have these columns. Its extra leagues (Argentina, Brazil, USA, Japan, Mexico,
   Nordics, Poland, Romania, Switzerland, Austria, Ireland, China) have results and odds
   only. Encode this per league as capability flags.
2. football-data.org v4. Free forever for 12 competitions: Champions League, Premier
   League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga,
   Championship, Brazilian Serie A, World Cup, European Championship. 10 requests per
   minute. Scores are delayed on free. Note that Europa League is NOT free here.
3. API-Football, 100 requests per day, every endpoint open. Use it only to fill gaps
   football-data.org leaves, notably Europa League and Conference League.
4. Understat (top 5 leagues plus Russia, from 2014/15) and FBref for expected goals.
   Scrape via the soccerdata Python package. Throttle FBref to one request per 4 seconds.
5. ClubElo, free, no key. Daily club Elo ratings, used as a strength prior.
6. The Odds API, 500 credits per month, for calibration benchmarking only.

Rate-limit rule: NEVER call an upstream provider on a web request. Scheduled workers
write to Postgres, the web tier reads only Postgres. One HTTP client with a token bucket,
exponential backoff on 429, ETag caching, circuit breaker, and a Redis daily budget
counter per source. When the budget runs out, serve cached predictions with a stale badge.

## The model

Layer 1, goal expectancy. Per league, fit team attack and defence plus home advantage:
  lambda_home = exp(mu + attack_home + defence_away + gamma)
  lambda_away = exp(mu + attack_away + defence_home)
Sum-to-zero constraints on attack and defence for identifiability.

Layer 2, Dixon-Coles correction. Independent Poisson under-predicts 0-0 and 1-1 and
over-predicts 1-0 and 0-1. Apply the tau correction with one dependence parameter rho to
exactly those four cells: tau = 1 - lambda*mu*rho at 0-0, 1 + lambda*rho at 0-1,
1 + mu*rho at 1-0, 1 - rho at 1-1, and 1 everywhere else. A fitted rho near -0.13 is normal.
Skipping this corrupts the draw, Under 2.5 and BTTS No.

Layer 3, time decay. Weight each historical match by exp(-xi * days_ago / 3.5). Start at
xi = 0.0065 per half-week (Dixon and Coles 1997, roughly a one-year half-life), then
re-optimise per league by maximising out-of-sample log-likelihood. Fit with
scipy.optimize.minimize using L-BFGS-B.

Layer 4, the grid. Build an 11x11 matrix of P(home=x, away=y) for 0..10, apply tau,
renormalise to sum to 1. Cache per fixture. Every goals market is a sum over this grid.

Layer 5, adjustments to lambda before the grid, each individually toggleable so its
contribution can be measured: blend goal-based and xG-based strengths (start 40/60);
shrink toward a ClubElo prior with weight proportional to 1/matches_played to handle
promoted teams and early season; rescale by league coefficient for cross-league European
competitions; rest days since last match; lineup-confirmed absences using minutes-weighted
xG plus xA share; a dead-rubber flag that widens the interval rather than shifting the mean.

Layer 6, corners and cards. Both are overdispersed relative to Poisson, so use negative
binomial with a league-fitted dispersion parameter. Corners: team corner-for and
corner-against rates, with shots and possession share as covariates, league mean around
10 to 11 total. Cards: the referee is the strongest single predictor, often stronger than
either team. Fit a referee effect from the Referee column, require 20 matches before a
referee gets their own estimate and shrink to the league mean otherwise. Model booking
points (yellow 10, red 25), not raw card counts, because that is how the market prices it.
Add a derby flag.

Layer 7, calibration. Raw probabilities are over-confident. Fit isotonic regression per
market on a rolling window of the last 2,000 settled predictions. Store versioned
calibration maps. Display calibrated values only. Publish a reliability diagram per market.

Layer 8, market anchor. Where odds exist, store the overround-stripped implied probability
beside the model probability. Do not blend. Surface the gap as a model-vs-market line.

Halves: fit separate first-half and second-half rates per league (around 45 percent of
goals come before the break), build two grids, combine assuming conditional independence.
First goal timing: inhomogeneous Poisson with piecewise-constant intensity in 15-minute
buckets.

## Market derivation, all from the grid
1X2 = sums of x>y, x=y, x<y. Double chance and draw-no-bet from those. Over/under totals =
sum cells where x+y crosses the line, for 0.5 through 5.5. BTTS = sum cells with x>0 and
y>0. Correct score = individual cells, top 12 plus other. Winning margin = anti-diagonals
of x-y. Odd/even = parity of x+y. Team totals = marginal Poisson on one lambda. Clean
sheet = the zero row or column. Win to nil = x>0 and y=0. European handicap = shift the
grid then resolve 1X2. Asian handicap from -2.5 to +2.5 in 0.25 steps, where quarter lines
split the stake across the two adjacent half lines and average. Corner and card markets
come from the negative binomial CDFs and the convolution of the two team marginals.

Display rules: render a market only if the league has the underlying data. Tag every
market with a quality badge (green for 3+ seasons of direct data, amber for 1-2 seasons or
cross-competition inference, grey for insufficient). Show probability as a whole percent
and fair odds as 1/p labelled as carrying no bookmaker margin. Never produce a best-bet
list or rank by expected profit; rank by model confidence only.

## Screens
/ Today: date strip, league filter chips, fixture cards with a stacked 1X2 bar, likely
scoreline and one highlighted market.
/league/[code]: fixtures, standings, season trend tiles versus the 5-season mean.
/match/[id]: header with referee and their cards-per-match; headline 1X2 bar, predicted
score, expected goals, confidence from the entropy of the 1X2 distribution; collapsible
market groups (Goals, BTTS, Result and handicaps, Corners, Cards, Halves, Correct score);
an 11x11 score-grid heatmap with the top three cells labelled; form and head to head with
goals, corners and cards per match; stat comparison bars per 90; model vs market; three
generated sentences naming the largest drivers; footer with as-of time, model version and
the responsible gambling notice.
/team/[id], /accuracy (public scorecard: Brier, log loss, hit rate, sample size,
reliability diagram), /how-it-works, /search.

Mobile-first. Server components for all data fetching. ISR revalidate 15 minutes, dropping
to 60 seconds inside 2 hours of kickoff. Lighthouse performance above 90 on simulated 4G,
LCP under 2.5s. Dark mode default. PWA offline shell. Keyboard-navigable accordions, ARIA
labels on probability bars, 4.5:1 contrast, no layout shift.

## Database
Tables: leagues (with capability flags), seasons, teams, team_aliases, referees, matches
(the fact table, unique on league + season + home team + away team), elo_ratings, model_runs (immutable,
one per fit), team_strengths, predictions (immutable, stores the compressed grid),
market_predictions (what the UI reads, indexed on match and market), calibration_maps,
outcomes, accuracy_metrics, ingest_runs.

Team name resolution is its own module with a persisted alias table. Every source spells
clubs differently. Fuzzy-match on first ingest, require manual confirmation below 0.92
similarity. Getting this wrong silently corrupts everything downstream.

All timestamps UTC in the database, rendered in the viewer timezone, defaulting to
Africa/Johannesburg.

Jobs: csv_backfill (weekly), fixtures_sync (daily 03:00 UTC), results_sync (hourly on
matchdays), xg_sync (weekly), elo_sync (daily), fit_models (nightly 04:00), predict (daily
05:00 and 90 minutes before kickoff), calibrate (weekly), score_accuracy (daily).
Backfill 5 seasons for fitting plus 2 for validation. Make the backfill an idempotent,
resumable CLI command.

## Validation
Walk-forward only. Never random k-fold: it leaks the future into the past and makes a
broken model look excellent. Primary metric is ranked probability score for 1X2, plus log
loss, plus Brier for binary markets.

Realistic targets. If the backtest beats these materially, assume leakage and find it:
1X2 RPS 0.195-0.205 (base-rate baseline 0.226); 1X2 log loss 1.00-1.03 (uniform 1.099);
1X2 hit rate 52-55 percent; over/under 2.5 log loss 0.66-0.68 (coin flip 0.693); BTTS the
same; corners Brier 0.225-0.240; cards Brier 0.230-0.245.

Leakage tests to write: every feature timestamp strictly before kickoff, asserted in the
feature builder; team strengths must come from a model_run fitted before the match date,
enforced by a database constraint; no season-aggregate features for matches inside that
season; backtests reproduce bit-identically given a seed.

Publication gates, all four required before a market renders: 500+ settled predictions;
calibration error under 5 percentage points in every decile; beats its naive baseline over
the last 1,000 settled predictions; the league has direct data or the cross-competition
flag is explicitly on.

## Engineering rules
The engine is a pure library: fitting functions take DataFrames and return parameters, with
no network or database calls inside model code. Write the test before the implementation
for anything in services/engine. Every prediction row records model version, parameter set
id, feature snapshot and timestamp, and is immutable. Every magic number lives in a
per-league config table, not inline. Never fabricate data: if a source is unreachable, fail
loudly and log to ingest_runs. Add a make verify target that fits one league, predicts one
fixture and prints the grid.

## How to proceed
Nine phases. Stop at the end of each, run the tests, summarise what changed and wait for me.
0. Scaffold: monorepo, compose, Postgres, Redis, Alembic, health checks, CI.
1. Ingest: CSV loader for the Premier League, football-data.org client, alias resolver.
2. Engine: Dixon-Coles with tau and decay, pure functions, unit tests.
3. Backtest: walk-forward harness against three baselines.
4. Markets: every goals-derived market with a golden-file test.
5. Corners and cards: negative binomial fits, referee effect, capability flags.
6. API and web: FastAPI endpoints plus Today, League and Match pages.
7. Calibration: isotonic maps, accuracy page, reliability diagrams.
8. Halves and timing.
9. Ship: full backfill, cron, deploy, monitoring.
