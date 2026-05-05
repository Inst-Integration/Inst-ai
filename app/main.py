from fastapi import FastAPI
from app.routers import transcription
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="inst-ai",
    description="악보 추출 파이프라인 API",
    version="0.1.0",
)

app.include_router(transcription.router)

@app.get("/health")
async def health():
    return {"status": "ok"}