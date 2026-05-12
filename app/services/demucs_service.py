import asyncio
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model
from scipy.signal import butter, sosfilt
import torchaudio
import torch

# 모듈 레벨에서 None으로 선언
_model = None
_device = None


def load_model():
    global _model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = get_model("htdemucs_ft")
    _model.eval()
    _model = _model.to(_device)
    return _model


def get_loaded_model():
    if _model is None:
        load_model()
    return _model


async def separate_audio(audio_path: str, instrument: str = "bass") -> str:
    loop = asyncio.get_running_loop()  # BUG-21: get_event_loop deprecated
    return await loop.run_in_executor(None, _separate, audio_path, instrument)


def _lowpass_filter(wav_numpy: np.ndarray, cutoff: int = 300, sr: int = 44100) -> np.ndarray:
    sos = butter(5, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, wav_numpy)


def _separate(audio_path: str, instrument: str) -> str:
    model = get_loaded_model()  # 전역 모델 재사용

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)

    wav = wav.to(_device)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav.unsqueeze(0),
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]

    track_index = 1 if instrument == "bass" else 2
    track = sources[track_index].cpu().numpy()

    if instrument == "bass":
        track = _lowpass_filter(track, cutoff=300, sr=44100)

    track_tensor = torch.from_numpy(track)

    max_val = track_tensor.abs().max()
    if max_val > 0:
        track_tensor = track_tensor / max_val * 0.9

    suffix = "_bass.wav" if instrument == "bass" else "_other.wav"
    output_path = audio_path.replace(".wav", suffix)
    torchaudio.save(output_path, track_tensor, 44100)

    return output_path