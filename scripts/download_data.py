"""Download PRC Data Challenge 2026 data from OpenSky MinIO (static access key)."""
import json
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
CRED = json.loads((ROOT / ".secrets" / "s3_credentials.json").read_text())
DATA_DIR = ROOT / "data"


def client():
    return boto3.client(
        "s3",
        endpoint_url=CRED["endpoint"],
        aws_access_key_id=CRED["access_key"],
        aws_secret_access_key=CRED["secret_key"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
    )


def main():
    s3 = client()
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    print("Buckets:", buckets)
    data_bucket = next(b for b in buckets if b != CRED["submission_bucket"])
    print("Data bucket:", data_bucket)

    DATA_DIR.mkdir(exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    objs = []
    for page in paginator.paginate(Bucket=data_bucket):
        objs.extend(page.get("Contents", []))
    print(f"{len(objs)} objects:")
    for o in objs:
        print(f" - {o['Key']:55s} {o['Size']/1e6:8.1f} MB")
    for o in objs:
        key = o["Key"]
        dest = DATA_DIR / key.replace("/", "_")
        if dest.exists() and dest.stat().st_size == o["Size"]:
            print(f"[skip] {key}")
            continue
        print(f"[dl]   {key}")
        s3.download_file(data_bucket, key, str(dest))
    print("Done.")


if __name__ == "__main__":
    main()
