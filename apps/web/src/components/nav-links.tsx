"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavLinksProps {
  leagues: { code: string; name: string }[];
}

const staticLinks = [
  { href: "/", label: "Home" },
  { href: "/accuracy", label: "Accuracy" },
  { href: "/how-it-works", label: "How It Works" },
];

export function NavLinks({ leagues }: NavLinksProps) {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);
  const [leagueOpen, setLeagueOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  function isActive(href: string) {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  }

  const leagueActive = pathname.startsWith("/league/");

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setLeagueOpen(false);
      }
    }
    if (leagueOpen) {
      document.addEventListener("mousedown", handleClick);
      return () => document.removeEventListener("mousedown", handleClick);
    }
  }, [leagueOpen]);

  // Close dropdown on route change
  useEffect(() => {
    setLeagueOpen(false);
    setMenuOpen(false);
  }, [pathname]);

  const chevron = (
    <svg className="w-3 h-3 ml-1" fill="none" viewBox="0 0 12 12" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5l3 3 3-3" />
    </svg>
  );

  return (
    <>
      {/* Desktop links */}
      <div className="hidden md:flex items-center gap-6">
        <Link
          href="/"
          className={`text-sm hover:text-zinc-200 ${
            isActive("/") ? "text-zinc-100" : "text-zinc-400"
          }`}
        >
          Home
        </Link>

        {/* Leagues dropdown */}
        <div ref={dropdownRef} className="relative">
          <button
            onClick={() => setLeagueOpen((v) => !v)}
            className={`text-sm hover:text-zinc-200 flex items-center ${
              leagueActive ? "text-zinc-100" : "text-zinc-400"
            }`}
          >
            Leagues
            {chevron}
          </button>
          {leagueOpen && (
            <div className="absolute top-8 left-0 min-w-[180px] bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl py-1 z-50">
              {leagues.map((l) => (
                <Link
                  key={l.code}
                  href={`/league/${l.code}`}
                  className={`block px-4 py-2 text-sm hover:bg-zinc-800 ${
                    pathname === `/league/${l.code}` ? "text-zinc-100 font-medium" : "text-zinc-400"
                  }`}
                >
                  {l.name}
                </Link>
              ))}
            </div>
          )}
        </div>

        <Link
          href="/accuracy"
          className={`text-sm hover:text-zinc-200 ${
            isActive("/accuracy") ? "text-zinc-100" : "text-zinc-400"
          }`}
        >
          Accuracy
        </Link>
        <Link
          href="/how-it-works"
          className={`text-sm hover:text-zinc-200 ${
            isActive("/how-it-works") ? "text-zinc-100" : "text-zinc-400"
          }`}
        >
          How It Works
        </Link>
      </div>

      {/* Mobile hamburger */}
      <button
        className="md:hidden ml-auto text-zinc-400 hover:text-zinc-200"
        onClick={() => setMenuOpen((v) => !v)}
        aria-label="Toggle menu"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          {menuOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="md:hidden absolute top-14 left-0 right-0 bg-zinc-950 border-b border-zinc-800 px-4 md:px-6 py-3 flex flex-col gap-1 z-50 overflow-x-hidden">
          {staticLinks.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={`text-sm py-2 hover:text-zinc-200 ${
                isActive(l.href) ? "text-zinc-100 font-medium" : "text-zinc-400"
              }`}
            >
              {l.label}
            </Link>
          ))}
          <div className="border-t border-zinc-800 mt-1 pt-1">
            <span className="text-xs text-zinc-600 uppercase tracking-wider px-0 py-1 block">
              Leagues
            </span>
            {leagues.map((l) => (
              <Link
                key={l.code}
                href={`/league/${l.code}`}
                className={`text-sm py-2 pl-2 block hover:text-zinc-200 ${
                  pathname === `/league/${l.code}` ? "text-zinc-100 font-medium" : "text-zinc-400"
                }`}
              >
                {l.name}
              </Link>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
