"""
Download the DNS-Challenge archive subset listed in configs/dns_subset.yaml.

Streams each archive to disk in chunks (never buffers a whole multi-GB file in
memory), skips archives that are already fully downloaded, and retries
transient failures a few times before giving up on a given file.

Usage:
    python scripts/download_dns_subset.py [--config configs/dns_subset.yaml]
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

CHUNK_SIZE = 1 << 20  # 1MB
MAX_RETRIES = 3


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None


def download_one(url: str, dest: Path) -> None:
    expected = remote_size(url)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        print(f"  already downloaded, skipping ({expected / 1e9:.2f} GB)")
        return

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    downloaded = tmp_dest.stat().st_size if tmp_dest.exists() else 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            if downloaded:
                req.add_header("Range", f"bytes={downloaded}-")
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(tmp_dest, "ab" if downloaded else "wb") as out:
                total = expected or 0
                last_print = time.monotonic()
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print > 5:
                        pct = f"{100 * downloaded / total:5.1f}%" if total else "?"
                        print(f"  {downloaded / 1e9:.2f} GB ({pct})", flush=True)
                        last_print = now
            tmp_dest.rename(dest)
            print(f"  done: {dest.stat().st_size / 1e9:.2f} GB")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dns_subset.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    base_url = config["base_url"].rstrip("/")
    raw_dir = Path(config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    blobs = config["clean_speech_archives"] + config["noise_archives"]
    print(f"Downloading {len(blobs)} archives to {raw_dir}/\n")

    for i, blob in enumerate(blobs, 1):
        url = f"{base_url}/{blob}"
        dest = raw_dir / Path(blob).name
        print(f"[{i}/{len(blobs)}] {blob}")
        download_one(url, dest)

    print("\nAll archives downloaded.")


if __name__ == "__main__":
    sys.exit(main())
