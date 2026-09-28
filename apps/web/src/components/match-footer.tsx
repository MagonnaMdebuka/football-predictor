/** Match page footer with model info and responsible gambling notice. */

export function MatchFooter() {
  return (
    <div className="mt-8 pt-6 border-t border-zinc-800 text-center text-xs text-zinc-600 space-y-2">
      <p>Model: Dixon-Coles v1.0.0</p>
      <p>
        Predictions are for entertainment and research purposes only.
        Please gamble responsibly. If you need help, visit{" "}
        <a
          href="https://www.begambleaware.org"
          target="_blank"
          rel="noopener noreferrer"
          className="text-zinc-400 underline hover:text-zinc-300"
        >
          BeGambleAware.org
        </a>
      </p>
    </div>
  );
}
