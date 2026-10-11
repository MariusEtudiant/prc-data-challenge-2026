"""Upload a submission parquet to the organisers' MinIO bucket, and read back its score.

The bucket is the only channel: the organisers score whatever lands in it and write a
`<file>_result.json` beside it. Five uploads per UTC day.

The file is checked before it goes anywhere - row count, column names, no missing values, and
the MVT_ID order against `submitting.parquet` - because a malformed file still spends one of
the five slots.

Usage: python scripts/submit.py list                 # what is already in the bucket
       python scripts/submit.py check  <file>        # validate without uploading
       python scripts/submit.py send   <file>        # validate, then upload
       python scripts/submit.py result <file>        # read the score back
"""
import json
import sys
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
CRED = json.loads((ROOT / ".secrets" / "s3_credentials.json").read_text())
N_ROWS = 344841


def client():
    return boto3.client("s3", endpoint_url=CRED["endpoint"],
                        aws_access_key_id=CRED["access_key"],
                        aws_secret_access_key=CRED["secret_key"],
                        config=Config(signature_version="s3v4"))


def check(path):
    p = Path(path)
    sub = pd.read_parquet(p)
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet")
    ok = True

    def say(cond, msg):
        nonlocal ok
        ok = ok and cond
        print("  %-4s %s" % ("OK" if cond else "NON", msg))

    say(list(sub.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
        "colonnes %s" % list(sub.columns))
    say(len(sub) == N_ROWS, "%d lignes (attendu %d)" % (len(sub), N_ROWS))
    say(sub["TAXITIME_SEC_mvt"].notna().all(), "aucune valeur manquante")
    say(bool((sub["TAXITIME_SEC_mvt"] >= 0).all()), "aucune valeur negative")
    say(bool((sub["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all()),
        "MVT_ID dans l'ordre du gabarit")
    say(p.name.startswith(CRED["team"]), "nom de fichier prefixe par l'equipe (%s)" % p.name)
    t = sub["TAXITIME_SEC_mvt"]
    print("  --   mediane %.0f s | moyenne %.0f | p99 %.0f | max %.0f" % (
        t.median(), t.mean(), t.quantile(.99), t.max()))
    return ok


def main():
    cmd = sys.argv[1]
    s3, b = client(), CRED["submission_bucket"]
    if cmd == "list":
        r = s3.list_objects_v2(Bucket=b).get("Contents", [])
        print("bucket %s : %d objets" % (b, len(r)))
        for o in sorted(r, key=lambda x: x["LastModified"]):
            print("  %-46s %10.1f Ko  %s" % (o["Key"], o["Size"] / 1024,
                                             o["LastModified"].strftime("%Y-%m-%d %H:%M")))
        return
    f = Path(sys.argv[2])
    if cmd == "check":
        print("verification de %s" % f.name)
        print("\n%s" % ("valide" if check(f) else "INVALIDE - rien n'est envoye"))
        return
    if cmd == "result":
        try:
            o = s3.get_object(Bucket=b, Key=f.name + "_result.json")
            print(json.dumps(json.loads(o["Body"].read()), indent=1))
        except Exception as e:
            print("pas encore de resultat (%s)" % type(e).__name__)
        return
    if cmd == "send":
        print("verification de %s" % f.name)
        if not check(f):
            print("\nINVALIDE - rien n'est envoye")
            sys.exit(1)
        s3.upload_file(str(f), b, f.name)
        print("\nenvoye dans %s/%s" % (b, f.name))
        print("le score arrive dans %s_result.json" % f.name)


if __name__ == "__main__":
    main()
