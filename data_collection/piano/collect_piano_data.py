"""
피아노 주법 데이터 수집 스크립트
inst-ai/data_collection/piano/collect_piano_data.py

사용법:
  python collect_piano_data.py --input trill.wav --label trill
  python collect_piano_data.py --input arpeggio.wav --label arpeggio
  python collect_piano_data.py --input normal.wav --label normal

베이스와 다른 점:
  - 세그먼트 길이 2초 (트릴/아르페지오 패턴 전체가 들어와야 함)
  - onset 기반으로 패턴 시작점을 잡고 2초 추출
  - 패턴과 패턴 사이 2초 이상 간격 권장
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf

SR = 44100
SEGMENT_DURATION = 2.0    # 베이스(0.5초)보다 길게. 패턴 전체 포함
MIN_ONSET_INTERVAL = 0.8  # 패턴 시작 간격 최소 0.8초
TOP_DB = 40


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=SR, mono=True)
    return audio, sr


def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    onset_times = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        backtrack=True,
        units='time',
    )
    return onset_times


def filter_onsets(onset_times: np.ndarray, min_interval: float) -> np.ndarray:
    if len(onset_times) == 0:
        return onset_times
    filtered = [onset_times[0]]
    for t in onset_times[1:]:
        if t - filtered[-1] >= min_interval:
            filtered.append(t)
    return np.array(filtered)


def extract_segments(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    segment_duration: float,
) -> list[np.ndarray]:
    segment_len = int(segment_duration * sr)
    segments = []

    for t in onset_times:
        start = int(t * sr)
        end = start + segment_len

        if end <= len(audio):
            seg = audio[start:end]
        else:
            seg = np.zeros(segment_len, dtype=np.float32)
            available = audio[start:]
            seg[:len(available)] = available

        if librosa.feature.rms(y=seg).mean() < librosa.db_to_amplitude(-TOP_DB):
            continue

        segments.append(seg)

    return segments


def save_segments(
    segments: list[np.ndarray],
    sr: int,
    label: str,
    output_dir: str,
) -> int:
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    existing = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
    start_idx = len(existing)

    for i, seg in enumerate(segments):
        filename = f"{label}_{start_idx + i:04d}.wav"
        filepath = os.path.join(label_dir, filename)
        sf.write(filepath, seg, sr)

    return len(segments)


def summarize(output_dir: str) -> None:
    print("\n── 현재 수집 현황 ──")
    for label in ['trill', 'arpeggio', 'normal']:
        label_dir = os.path.join(output_dir, label)
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith('.wav')])
            status = "✅" if count >= 200 else f"({count}/200)"
            print(f"  {label:10s}: {count:3d}개 {status}")
        else:
            print(f"  {label:10s}:   0개 (0/200)")
    print()


def main():
    parser = argparse.ArgumentParser(description="피아노 주법 데이터 수집")
    parser.add_argument('--input', required=True, help="입력 wav 파일 경로")
    parser.add_argument('--label', required=True,
                        choices=['trill', 'arpeggio', 'normal'],
                        help="주법 레이블")
    parser.add_argument('--output_dir', default='./data/raw',
                        help="세그먼트 저장 디렉토리 (기본: data_collection/piano/data/raw)")
    parser.add_argument('--segment_duration', type=float, default=SEGMENT_DURATION)
    parser.add_argument('--min_interval', type=float, default=MIN_ONSET_INTERVAL)
    args = parser.parse_args()

    print(f"입력: {args.input}")
    print(f"레이블: {args.label}")

    audio, sr = load_audio(args.input)
    print(f"오디오 로드 완료: {len(audio) / sr:.1f}초")

    onset_times = detect_onsets(audio, sr)
    print(f"onset 감지: {len(onset_times)}개")

    onset_times = filter_onsets(onset_times, args.min_interval)
    print(f"필터링 후: {len(onset_times)}개")

    segments = extract_segments(audio, sr, onset_times, args.segment_duration)
    print(f"유효 세그먼트: {len(segments)}개")

    saved = save_segments(segments, sr, args.label, args.output_dir)
    print(f"저장 완료: {args.output_dir}/{args.label}/")

    summarize(args.output_dir)


if __name__ == '__main__':
    main()