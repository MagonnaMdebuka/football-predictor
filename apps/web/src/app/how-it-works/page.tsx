import Link from "next/link";

export default function HowItWorksPage() {
  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-zinc-100">How It Works</h1>
        <p className="text-sm text-zinc-500 mt-1">
          A plain-English guide to the model, its data, and its limits
        </p>
      </div>

      <div className="space-y-6">
        {/* The Model */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">The Model</h2>
          </div>
          <div className="px-4 py-4 space-y-3 text-sm text-zinc-400 leading-relaxed">
            <p>
              At its core, the engine uses the{" "}
              <strong className="text-zinc-300">Dixon-Coles model</strong> — a
              well-established statistical approach to football prediction first
              published in 1997. The idea is straightforward: every team has an{" "}
              <strong className="text-zinc-300">attack strength</strong> and a{" "}
              <strong className="text-zinc-300">defence strength</strong>.
              A strong attack paired against a weak defence produces more
              expected goals, and vice versa.
            </p>
            <p>
              The model also accounts for{" "}
              <strong className="text-zinc-300">home advantage</strong> — the
              well-documented tendency for home teams to perform better — and
              applies a{" "}
              <strong className="text-zinc-300">low-score correction</strong>{" "}
              that improves accuracy for scorelines like 0-0, 1-0, 0-1, and 1-1,
              which pure Poisson models tend to get wrong.
            </p>
            <p>
              Recent matches count more than older ones through{" "}
              <strong className="text-zinc-300">time-decay weighting</strong>.
              A match from last month carries far more weight than one from two
              seasons ago, so the model adapts as teams improve or decline.
            </p>
            <p>
              From these strengths, the engine builds a{" "}
              <strong className="text-zinc-300">score grid</strong> — a table of
              probabilities for every possible scoreline (0-0 through 10-10).
              All market probabilities (1X2, over/under, BTTS, correct score)
              are derived directly from this grid by summing the relevant cells.
            </p>
          </div>
        </section>

        {/* Data Sources */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">Data Sources</h2>
          </div>
          <div className="px-4 py-4 space-y-3 text-sm text-zinc-400 leading-relaxed">
            <p>
              <strong className="text-zinc-300">Historical results</strong> come
              from{" "}
              <a
                href="https://www.football-data.co.uk"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300"
              >
                football-data.co.uk
              </a>
              , which provides CSV files covering 5+ seasons of match results,
              half-time scores, corners, cards, and closing odds for major
              European leagues. This data is used to fit the model parameters.
            </p>
            <p>
              <strong className="text-zinc-300">Upcoming fixtures</strong> are
              sourced from{" "}
              <a
                href="https://www.football-data.org"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300"
              >
                football-data.org
              </a>
              , which provides a structured API of scheduled matches. Fixtures
              are synced daily so predictions stay current.
            </p>
            <p>
              All data processing happens on a scheduled worker — the website
              itself only reads from the database, so predictions are never
              delayed by data fetching.
            </p>
          </div>
        </section>

        {/* Markets */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">Markets</h2>
          </div>
          <div className="px-4 py-4 space-y-3 text-sm text-zinc-400 leading-relaxed">
            <p>The model produces probabilities for the following markets:</p>
            <ul className="list-disc list-inside space-y-1 text-zinc-400">
              <li><strong className="text-zinc-300">Match Result (1X2)</strong> — home win, draw, or away win</li>
              <li><strong className="text-zinc-300">Over/Under Goals</strong> — 1.5, 2.5, and 3.5 goal lines</li>
              <li><strong className="text-zinc-300">Both Teams to Score</strong> — yes or no</li>
              <li><strong className="text-zinc-300">Correct Score</strong> — exact scoreline probabilities</li>
              <li><strong className="text-zinc-300">Half-Time Markets</strong> — HT result, HT over/under, HT BTTS</li>
              <li><strong className="text-zinc-300">HT/FT Double</strong> — combined half-time and full-time result (9 outcomes)</li>
              <li><strong className="text-zinc-300">First Goal Timing</strong> — probability of a goal before a given minute</li>
            </ul>
            <p>
              Each probability can be converted to a{" "}
              <strong className="text-zinc-300">fair odd</strong> by taking its
              inverse (e.g. a 40% chance = 2.50 fair odds). The difference
              between the model&apos;s fair odds and a bookmaker&apos;s offered
              odds is what identifies potential value.
            </p>
          </div>
        </section>

        {/* Accuracy */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">Accuracy</h2>
          </div>
          <div className="px-4 py-4 space-y-3 text-sm text-zinc-400 leading-relaxed">
            <p>
              Every prediction is backtested using a{" "}
              <strong className="text-zinc-300">walk-forward</strong> approach:
              the model is trained only on past data and then evaluated on
              matches it has never seen. This prevents overfitting and gives an
              honest measure of real-world performance.
            </p>
            <p>
              After backtesting, the model is{" "}
              <strong className="text-zinc-300">calibrated</strong> — its raw
              probabilities are adjusted so that events predicted at 30% really
              do happen about 30% of the time. Markets must pass{" "}
              <strong className="text-zinc-300">publication gates</strong>{" "}
              (minimum sample size, accuracy thresholds, must beat a naive
              baseline) before their predictions go live.
            </p>
            <p>
              Accuracy is measured using Brier score (for two-outcome markets)
              and Ranked Probability Score (for three-outcome markets like 1X2)
              — both standard metrics in probability forecasting. Lower is
              better for both. Full details are on the{" "}
              <Link
                href="/accuracy"
                className="text-blue-400 hover:text-blue-300"
              >
                Accuracy page
              </Link>
              .
            </p>
          </div>
        </section>

        {/* What It Cannot Do */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">What It Cannot Do</h2>
          </div>
          <div className="px-4 py-4 space-y-3 text-sm text-zinc-400 leading-relaxed">
            <p>
              The model is purely statistical and works from historical match
              results. It does <strong className="text-zinc-300">not</strong>{" "}
              incorporate:
            </p>
            <ul className="list-disc list-inside space-y-1 text-zinc-400">
              <li>Live or in-play data — predictions are pre-match only</li>
              <li>Player-level information — no lineups, form, or fitness</li>
              <li>Injuries, suspensions, or squad rotation</li>
              <li>Weather conditions or pitch state</li>
              <li>Managerial changes or tactical shifts</li>
              <li>Motivation factors (relegation battles, title races, dead rubbers)</li>
            </ul>
            <p>
              Probabilities are estimates based on historical patterns, not
              certainties. A 70% probability still means the outcome fails to
              happen three times in ten.
            </p>
          </div>
        </section>

        {/* Responsible Gambling */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">Responsible Gambling</h2>
          </div>
          <div className="px-4 py-4 space-y-3 text-sm text-zinc-400 leading-relaxed">
            <p>
              The predictions on this site are provided for{" "}
              <strong className="text-zinc-300">
                entertainment and research purposes only
              </strong>
              . They do not constitute financial advice, betting tips, or a
              recommendation to gamble. No model can guarantee profit.
            </p>
            <p>
              If you choose to gamble, only bet what you can afford to lose.
              Set limits, take breaks, and seek help if gambling stops being fun.
            </p>
            <p>Support is available from:</p>
            <ul className="list-disc list-inside space-y-1 text-zinc-400">
              <li>
                <a
                  href="https://www.begambleaware.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300"
                >
                  BeGambleAware
                </a>{" "}
                — free, confidential advice and support
              </li>
              <li>
                <a
                  href="https://www.gamcare.org.uk"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300"
                >
                  GamCare
                </a>{" "}
                — counselling and treatment for problem gambling
              </li>
              <li>
                <strong className="text-zinc-300">
                  National Gambling Helpline
                </strong>{" "}
                — 0808 8020 133 (free, 24/7)
              </li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
