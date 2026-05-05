from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.yt_service import download_audio
from app.services.demucs_service import separate_audio
from app.services.pitch_service import audio_to_musicxml
from app.services.s3_service import upload_and_get_url

router = APIRouter(prefix="/transcription", tags=["transcription"])

class TranscriptionRequest(BaseModel):
    instrument: str = "bass"  # "bass" 또는 "piano"
    youtube_url: str

class TranscriptionResponse(BaseModel):
    musicxml_url: str  # S3 presigned URL
    expires_in: int    # 유효 시간 (초)

@router.post("/", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest):
    try:
        # 1. 유튜브 오디오 추출
        audio_path = await download_audio(request.youtube_url)

        # 2. Demucs로 음원 분리 (악기별 트랙)
        melody_path = await separate_audio(audio_path, request.instrument)

        # 3. Basic-Pitch로 음정 인식 → MusicXML
        musicxml_path = await audio_to_musicxml(melody_path, request.instrument)

        # 4. S3 임시 저장 → presigned URL
        url, expires_in = await upload_and_get_url(musicxml_path)

        return TranscriptionResponse(
            musicxml_url=url,
            expires_in=expires_in,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))