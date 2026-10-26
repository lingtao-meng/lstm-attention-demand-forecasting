#!/usr/bin/env python3
"""Download the M5 files with parallel HTTP range requests.

The public Hugging Face mirror is throttled per connection; a single curl gets
~100 KB/s.  Splitting the file into 8 MB ranges and running 12 connections is
much faster and still uses the same public files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://huggingface.co/datasets/denephew/M5_Forecasting/resolve/main"
FILES = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]


def remote_size(url: str) -> tuple[int, bool]:
    headers = subprocess.check_output(["curl", "-sIL", url], text=True, timeout=60)
    size = 0
    supports_ranges = False
    for line in headers.splitlines():
        key, _, value = line.partition(":")
        if key.lower() == "content-length" and value.strip().isdigit():
            size = int(value.strip())
        if key.lower() == "accept-ranges" and "bytes" in value.lower():
            supports_ranges = True
    return size, supports_ranges


def fetch(url: str, start: int, end: int, path: Path, attempts: int = 5) -> None:
    expected = end - start + 1
    for attempt in range(attempts):
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3", "--retry-all-errors", "-sS",
             "-r", f"{start}-{end}", url, "-o", str(path)],
            check=False,
        )
        if path.exists() and path.stat().st_size == expected:
            return
    raise RuntimeError(f"range {start}-{end} failed after {attempts} attempts")


def download(url: str, out: Path, workers: int = 12, chunk_mb: int = 8) -> None:
    size, supports_ranges = remote_size(url)
    if not supports_ranges or size < chunk_mb * 1024 * 1024:
        subprocess.run(["curl", "-L", "--fail", "-sS", url, "-o", str(out)], check=True)
        return
    chunk = chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk - 1, size - 1)) for start in range(0, size, chunk)]
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".parts-") as tmp:
        paths = [Path(tmp) / f"part_{i:04d}" for i in range(len(ranges))]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, url, start, end, path) for (start, end), path in zip(ranges, paths)]  # noqa: B905
            for future in futures:
                future.result()
        with open(out, "wb") as target:
            for path in paths:
                with open(path, "rb") as source:
                    shutil.copyfileobj(source, target)
    actual = out.stat().st_size
    if actual != size:
        raise RuntimeError(f"{out.name}: expected {size} bytes, got {actual}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = data_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"skip {name} ({target.stat().st_size:,} bytes)")
            continue
        print(f"download {name}")
        download(f"{BASE}/{name}", target, workers=args.workers)
        print(f"  {target.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
