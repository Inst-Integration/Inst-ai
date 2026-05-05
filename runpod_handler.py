import asyncio
import runpod
from app.services.demucs_service import load_model as load_demucs
from app.services.pitch_service import load_model as load_pitch
from app.services.demucs_service import separate_audio
from app.services.pitch_service import audio_to_musicxml
from app.services.yt_service import download_audio
from app.services.s3_service import upload_and_get_url

# 워커 시작 시 1회 — 재사용 시 모델 재로드 없음
load_demucs()
load_pitch()


def handler(job):
    input_data = job["input"]
    youtube_url = input_data.get("youtube_url")
    instrument = input_data.get("instrument", "bass")

    if not youtube_url:
        return {"error": "youtube_url required"}

    loop = asyncio.new_event_loop()
    try:
        audio_path = loop.run_until_complete(download_audio(youtube_url))
        bass_path  = loop.run_until_complete(separate_audio(audio_path, instrument))
        xml_path   = loop.run_until_complete(audio_to_musicxml(bass_path, instrument))
        url, expires = loop.run_until_complete(upload_and_get_url(xml_path))
        return {"musicxml_url": url, "expires_in": expires}
    except Exception as e:
        return {"error": str(e)}
    finally:
        loop.close()


runpod.serverless.start({"handler": handler})