"""Print the submission bucket content; used to poll for result files."""
import json
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
cred = json.loads((ROOT / ".secrets" / "s3_credentials.json").read_text())
s3 = boto3.client("s3", endpoint_url=cred["endpoint"],
                  aws_access_key_id=cred["access_key"],
                  aws_secret_access_key=cred["secret_key"],
                  config=Config(signature_version="s3v4", retries={"max_attempts": 5}))
for o in s3.list_objects_v2(Bucket=cred["submission_bucket"]).get("Contents", []):
    print(f"{o['Key']}\t{o['Size']}")
