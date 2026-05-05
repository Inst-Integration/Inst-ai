import boto3
import os
import uuid
from datetime import datetime
import asyncio

AWS_BUCKET = os.getenv("AWS_BUCKET_NAME", "inst-musicxml")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PRESIGNED_URL_EXPIRY = 600

_s3 = None

def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=AWS_REGION)
    return _s3


def _upload(file_path: str) -> str:
    s3 = _get_s3()
    key = f"musicxml/{datetime.utcnow().strftime('%Y%m%d')}/{uuid.uuid4()}.xml"
    s3.upload_file(file_path, AWS_BUCKET, key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )
    return url


async def upload_and_get_url(file_path: str) -> tuple[str, int]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일 없음: {file_path}")
    loop = asyncio.get_running_loop()
    url = await loop.run_in_executor(None, _upload, file_path)
    return url, PRESIGNED_URL_EXPIRY