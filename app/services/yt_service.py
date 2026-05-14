import asyncio
import base64
import os
import tempfile
import yt_dlp

# RunPod 환경변수 YOUTUBE_COOKIES_B64에 base64 인코딩된 cookies.txt를 넣으면
# 워커 시작 시 /tmp/yt_cookies.txt로 복원해서 yt-dlp에 전달
_COOKIES_FILE: str | None = None

def _init_cookies() -> str | None:
    b64 = os.getenv("YOUTUBE_COOKIES_B64", "").strip()
    if not b64:
        return None
    try:
        path = "/tmp/yt_cookies.txt"
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return path
    except Exception as e:
        print(f"[yt_service] 쿠키 초기화 실패: {e}")
        return None

_COOKIES_FILE = _init_cookies()


async def download_audio(youtube_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "audio")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        "socket_timeout": 30,
        "retries": 3,
        "quiet": True,
        "no_warnings": True,
    }

    if _COOKIES_FILE:
        ydl_opts["cookiefile"] = _COOKIES_FILE

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _download, ydl_opts, youtube_url)

    return output_path + ".wav"


def _download(ydl_opts: dict, url: str):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
