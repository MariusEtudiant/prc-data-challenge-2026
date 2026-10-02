"""Open a Trino session (browser OAuth) and cache the token for later queries."""
import warnings, sys
warnings.filterwarnings("ignore")
from pyopensky.trino import Trino

t = Trino()
print("=== running test query ===", flush=True)
try:
    df = t.query("SELECT 1 AS ok")
    print("TRINO OK:", df.to_dict(), flush=True)
    df2 = t.query("SHOW TABLES FROM minio.osky")
    print("TABLES:", df2.head(30).to_string(), flush=True)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:400], flush=True)
