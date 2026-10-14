"""Parallel range downloader for slow single-stream hosts (GitHub release assets).

  python -m pipeline.fetch_url <dest_dir> <url> [<url> ...]
Skips files already complete; writes <name>.part then renames. Prints one line per file.
"""
import concurrent.futures as cf
import os
import sys
import time

import requests

CHUNK = 16 << 20


def size_of(url: str) -> tuple[int, str]:
    r = requests.get(url, stream=True, allow_redirects=True, timeout=60, headers={"Range": "bytes=0-0"})
    final = r.url
    cr = r.headers.get("Content-Range", "")
    r.close()
    return (int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("Content-Length", 0))), final


def fetch(url: str, dest_dir: str, streams: int = 12) -> str:
    name = url.rsplit("/", 1)[-1].split("?")[0]
    dest = os.path.join(dest_dir, name)
    total, final = size_of(url)
    if os.path.exists(dest) and (total == 0 or os.path.getsize(dest) == total):
        return f"skip {name}"
    part = dest + ".part"
    t0 = time.time()
    if total < 4 * CHUNK:
        with requests.get(final, timeout=600) as r:
            r.raise_for_status()
            open(part, "wb").write(r.content)
        total = 0  # small files may arrive gzip-encoded; Content-Length is then the compressed size
    else:
        with open(part, "wb") as f:
            f.truncate(total)

        def get(rng):
            a, b = rng
            for attempt in range(5):
                try:
                    with requests.get(final, headers={"Range": f"bytes={a}-{b}"}, timeout=120) as r:
                        r.raise_for_status()
                        with open(part, "r+b") as f:
                            f.seek(a)
                            f.write(r.content)
                    return
                except requests.RequestException:
                    time.sleep(3 * (attempt + 1))
            raise RuntimeError(f"range {a}-{b} failed")

        ranges = [(a, min(a + CHUNK, total) - 1) for a in range(0, total, CHUNK)]
        with cf.ThreadPoolExecutor(streams) as ex:
            list(ex.map(get, ranges))
    if total and os.path.getsize(part) != total:
        raise RuntimeError(f"{name}: size {os.path.getsize(part)} != {total}")
    os.replace(part, dest)
    dt = time.time() - t0
    return f"ok   {name} {total / 1e6:.0f} MB in {dt:.0f}s ({total / max(dt, 1e-3) / 1e6:.1f} MB/s)"


if __name__ == "__main__":
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    for u in sys.argv[2:]:
        print(fetch(u, d), flush=True)
