import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Football Predictor",
  description: "Football match prediction platform powered by Dixon-Coles",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 min-h-screen flex flex-col">
        <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-5xl mx-auto px-4 h-14 flex items-center gap-6">
            <Link href="/" className="text-lg font-bold text-zinc-100 hover:text-white">
              Football Predictor
            </Link>
            <Link href="/" className="text-sm text-zinc-400 hover:text-zinc-200">
              Home
            </Link>
            <Link href="/league/E0" className="text-sm text-zinc-400 hover:text-zinc-200">
              Leagues
            </Link>
            <Link href="/accuracy" className="text-sm text-zinc-400 hover:text-zinc-200">
              Accuracy
            </Link>
            <Link href="/how-it-works" className="text-sm text-zinc-400 hover:text-zinc-200">
              How It Works
            </Link>
          </div>
        </nav>

        <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-6">
          {children}
        </main>

        <footer className="border-t border-zinc-800 py-6 text-center text-xs text-zinc-600">
          <p>Dixon-Coles v1.0.0 — Predictions for entertainment and research only</p>
          <p className="mt-1">
            Please gamble responsibly.{" "}
            <a
              href="https://www.begambleaware.org"
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-500 underline hover:text-zinc-400"
            >
              BeGambleAware.org
            </a>
          </p>
        </footer>
      </body>
    </html>
  );
}
