"""Fetch METAR history (Iowa Mesonet ASOS archive) for the 10 airports,
2025-01-01 .. 2026-08-01, save to data/external/metar.parquet."""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import AIRPORTS

OUT = ROOT / "data" / "external"
OUT.mkdir(parents=True, exist_ok=True)

URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
       "?station={st}&data=tmpf,dwpf,drct,sknt,gust,vsby,wxcodes,skyc1,skyl1"
       "&year1=2025&month1=1&day1=1&year2=2026&month2=8&day2=1"
       "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty&trace=T")

frames = []
for st in AIRPORTS:
    for attempt in range(3):
        try:
            r = requests.get(URL.format(st=st), timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if len(df) < 100:
                raise RuntimeError(f"suspiciously few rows: {len(df)}")
            frames.append(df)
            print(f"{st}: {len(df):,} obs  {df['valid'].min()} .. {df['valid'].max()}")
            break
        except Exception as e:
            print(f"{st} attempt {attempt+1} failed: {e}")
            time.sleep(10)
    else:
        raise SystemExit(f"failed to fetch {st}")
    time.sleep(2)

met = pd.concat(frames, ignore_index=True)
met["valid"] = pd.to_datetime(met["valid"], utc=True)
met.to_parquet(OUT / "metar.parquet", index=False)
print(f"\nsaved {len(met):,} rows -> {OUT / 'metar.parquet'}")
