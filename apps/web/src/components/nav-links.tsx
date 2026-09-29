"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/league/E0", label: "Premier League" },
  { href: "/accuracy", label: "Accuracy" },
  { href: "/how-it-works", label: "How It Works" },
];

export function NavLinks() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  function isActive(href: string) {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  }

  return (
    <>
      {/* Desktop links */}
      <div className="hidden md:flex items-center gap-6">
        {links.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={`text-sm hover:text-zinc-200 ${
              isActive(l.href) ? "text-zinc-100" : "text-zinc-400"
            }`}
          >
            {l.label}
          </Link>
        ))}
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
        <div className="md:hidden absolute top-14 left-0 right-0 bg-zinc-950 border-b border-zinc-800 px-4 py-3 flex flex-col gap-3 z-50">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              onClick={() => setMenuOpen(false)}
              className={`text-sm hover:text-zinc-200 ${
                isActive(l.href) ? "text-zinc-100" : "text-zinc-400"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </div>
      )}
    </>
  );
}
