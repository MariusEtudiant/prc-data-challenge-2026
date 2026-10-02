"""Driver: runs m6_staged.py stages sequentially, each in a fresh process."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stages = sys.argv[1:] or ["build", "lgbm", "cb", "blend"]
for s in stages:
    print(f"===== stage {s} =====", flush=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "m6_staged.py"), s],
                       cwd=str(ROOT))
    if r.returncode != 0:
        print(f"stage {s} FAILED with code {r.returncode}", flush=True)
        sys.exit(r.returncode)
print("ALL STAGES DONE", flush=True)
