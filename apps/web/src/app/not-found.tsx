import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
      <h1 className="text-5xl sm:text-6xl font-bold text-zinc-400 mb-4">404</h1>
      <p className="text-lg sm:text-xl text-zinc-300 mb-2">Page not found</p>
      <p className="text-sm text-zinc-500 mb-8 max-w-md">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link
        href="/"
        className="text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-100 px-4 py-2 rounded-lg transition-colors"
      >
        Back to Home
      </Link>
    </div>
  );
}
