import asyncio
import os
import tempfile
import yt_dlp


async def download_audio(youtube_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "audio")

    ydl_opts = {
        "format": "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best",
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

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _download, ydl_opts, youtube_url)

    return output_path + ".wav"


def _download(ydl_opts: dict, url: str):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])