"""Seed the running API's contract library with real CUAD contracts (dev/demo).

The API image doesn't ship the gold corpus, so this runs on the host against a
running API:  python seed_library.py [N]   (default 6; API_URL env or localhost:8000)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")
GOLD = Path(__file__).resolve().parent / "evals" / "gold" / "contracts"


def title_for(path: Path) -> str:
    parts = path.stem.split("_")
    name = f"{parts[0]} - {parts[-1]}" if len(parts) > 1 else path.stem
    return name[:70]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    files = sorted(GOLD.glob("*.txt"), key=lambda p: p.stat().st_size)[:n]  # smallest first
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        resp = httpx.post(
            f"{API_URL}/contracts",
            json={"title": title_for(path), "document_text": text},
            timeout=180,
        )
        meta = resp.json() if resp.status_code == 200 else resp.text[:120]
        print(f"  [{resp.status_code}] {title_for(path)!r} -> {meta}")
    listed = httpx.get(f"{API_URL}/contracts", timeout=30).json()
    print(f"\nlibrary now holds {len(listed)} contract(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
