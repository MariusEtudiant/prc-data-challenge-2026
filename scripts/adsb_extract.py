"""Extract ground movement at the ten challenge airports from adsb.lol globe_history.

Open data under ODbL, which the organiser confirmed is allowed ("the rule about open
data applies: anything you think that helps, use it"). Trino is not allowed; this is.

We predict off-block, and Network Manager's AOBT_3 - the input our whole model leans
on - is wrong by more than a minute four times out of five (sd 355s, and 455s at Rome)
because nobody observes pushback. ADS-B does observe it: a trace carries explicit
'ground' points, so the moment an aircraft starts moving from its stand is directly
measurable.

Each day is a ~3.6 GB uncompressed tar of gzipped per-aircraft JSON traces
(./traces/<xx>/trace_full_<icao24>.json). We stream it, keep only points inside the
airport boxes, and write one compact parquet per day. Nothing is kept on disk beyond
the day being processed.

Usage: python scripts/adsb_extract.py YYYY-MM-DD [YYYY-MM-DD ...]
"""
import gzip
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "adsb"
OUT.mkdir(parents=True, exist_ok=True)

# generous boxes: ~0.25 deg lat, scaled in lon, enough for the whole surface and approach
APT = {"EDDF": (50.033, 8.571), "EDDM": (48.354, 11.786), "EGLL": (51.470, -0.454),
       "EHAM": (52.309, 4.764), "LEBL": (41.297, 2.078), "LEMD": (40.472, -3.561),
       "LFPG": (49.010, 2.548), "LIRF": (41.800, 12.239), "LSZH": (47.458, 8.548),
       "LTFM": (41.262, 28.742)}
PAD_LAT = 0.09
TAGS = ["planes-readsb-prod-0", "planes-readsb-staging-0",
        "planes-readsb-prod-0tmp", "planes-readsb-staging-0tmp"]


def _boxes():
    out = {}
    for a, (la, lo) in APT.items():
        pad_lo = PAD_LAT / np.cos(np.radians(la))
        out[a] = (la - PAD_LAT, la + PAD_LAT, lo - pad_lo, lo + pad_lo)
    return out


BOX = _boxes()


def _which(lat, lon):
    for a, (la0, la1, lo0, lo1) in BOX.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return a
    return None


def _url(day, tag, part):
    """Release tags use dots, not dashes: v2025.07.15-planes-readsb-prod-0."""
    year = day[:4]
    dd = day.replace("-", ".")
    base = "https://github.com/adsblol/globe_history_%s/releases/download" % year
    name = "v%s-%s.tar%s" % (dd, tag, part)
    return "%s/v%s-%s/%s" % (base, dd, tag, name)


def _probe(url):
    try:
        r = urllib.request.Request(url, headers={"User-Agent": "prc-dc2026"}, method="HEAD")
        urllib.request.urlopen(r, timeout=60).close()
        return True
    except Exception:
        return False


def _open_stream(day):
    """Find the release tag that exists and list its parts, WITHOUT opening them.
    Connections are opened one at a time by _Chain: holding a second 2 GB download
    idle for the minutes the first one takes makes the server drop it, which showed
    up as 'unexpected end of data' midway through the tar."""
    for tag in TAGS:
        urls = [u for u in (_url(day, tag, sfx) for sfx in (".aa", ".ab", ".ac")) if _probe(u)]
        if urls:
            return tag, _Chain(urls)
        u = _url(day, tag, "")
        if _probe(u):
            return tag, _Chain([u])
    raise RuntimeError("aucune release trouvee pour %s" % day)


class _Chain(io.RawIOBase):
    """Several HTTP parts read as one stream, each opened only when reached.
    BufferedReader calls readinto, not read, so readinto is what must be provided."""

    def __init__(self, urls):
        self.urls = urls
        self.i = 0
        self.cur = None

    def readable(self):
        return True

    def _ensure(self):
        if self.cur is None and self.i < len(self.urls):
            r = urllib.request.Request(self.urls[self.i],
                                       headers={"User-Agent": "prc-dc2026"})
            self.cur = urllib.request.urlopen(r, timeout=180)

    def readinto(self, b):
        while self.i < len(self.urls):
            self._ensure()
            n = self.cur.readinto(b)
            if n:
                return n
            try:
                self.cur.close()
            except Exception:
                pass
            self.cur = None
            self.i += 1
        return 0


def extract_day(day):
    dst = OUT / ("ground_%s.parquet" % day)
    if dst.exists():
        print("  %s deja fait" % day, flush=True)
        return
    t0 = time.time()
    tag, stream = _open_stream(day)
    rows = []
    n_tr = n_kept = 0
    with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 22), mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".json"):
                continue
            n_tr += 1
            try:
                raw = tf.extractfile(m).read()
                d = json.loads(gzip.decompress(raw))
            except Exception:
                continue
            tr = d.get("trace") or []
            if not tr:
                continue
            base = d.get("timestamp", 0.0)
            icao = d.get("icao", "")
            reg = d.get("r", "")
            typ = d.get("t", "")
            hit = None
            for p in tr:
                a = _which(p[1], p[2])
                if a is not None:
                    hit = a
                    break
            if hit is None:
                continue
            n_kept += 1
            for p in tr:
                a = _which(p[1], p[2])
                if a is None:
                    continue
                onground = p[3] == "ground"
                cs = ""
                if len(p) > 8 and isinstance(p[8], dict):
                    cs = (p[8].get("flight") or "").strip()
                rows.append((icao, reg, typ, a, base + p[0], p[1], p[2],
                             1 if onground else 0,
                             float(p[4]) if p[4] is not None else np.nan, cs))
    df = pd.DataFrame(rows, columns=["icao24", "reg", "actype", "airport", "ts",
                                     "lat", "lon", "on_ground", "gs", "callsign"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df.to_parquet(dst, index=False)
    print("  %s [%s] %d traces lues, %d retenues, %d points -> %.1f Mo (%.1f min)" % (
        day, tag, n_tr, n_kept, len(df), dst.stat().st_size / 1e6, (time.time() - t0) / 60),
        flush=True)


if __name__ == "__main__":
    for day in sys.argv[1:]:
        try:
            extract_day(day)
        except Exception as e:
            print("  %s ECHEC: %s" % (day, e), flush=True)
