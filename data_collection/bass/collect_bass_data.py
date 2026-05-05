"""
베이스 주법 데이터 수집 스크립트
inst-ai/data_collection/collect_bass_data.py

사용법:
  python collect_bass_data.py --input slap.wav --label slap
  python collect_bass_data.py --input pop.wav --label pop
  python collect_bass_data.py --input finger.wav --label finger

녹음 가이드라인:
  - 음 하나 치고 1초 이상 쉬고 다음 음
  - E1, A1, D2, G2 골고루
  - 약하게 / 보통 / 강하게 세기 혼합
  - 레가토 + 스타카토 혼합 (레이블은 주법 기준으로만)
  - 목표: 주법별 100개 이상
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf


# ── 파라미터 ──────────────────────────────────────────────
SR = 44100          # 샘플레이트. inst DSP와 동일
SEGMENT_DURATION = 0.5   # onset 이후 몇 초를 한 샘플로 자를지 (attack + 잔향 포함)
MIN_ONSET_INTERVAL = 0.3 # 이보다 짧은 간격의 onset은 과감지로 판단해 병합 (초)
TOP_DB = 40         # librosa 묵음 구간 판단 기준 (dB). 환경 노이즈에 따라 조정 가능


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """wav 파일 로드. librosa는 기본 mono, float32로 반환"""
    audio, sr = librosa.load(path, sr=SR, mono=True)
    return audio, sr


def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    onset 감지.

    librosa.onset.onset_detect는 내부적으로 spectral flux 기반으로 동작.
    베이스 attack이 명확한 환경(1초 간격 연주)에서는 충분히 신뢰할 수 있음.

    backtrack=True: onset을 에너지 피크가 아닌 실제 음 시작점으로 보정.
    units='time': 샘플 인덱스가 아닌 초 단위로 반환.
    """
    onset_times = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        backtrack=True,
        units='time',
    )
    return onset_times


def filter_onsets(onset_times: np.ndarray, min_interval: float) -> np.ndarray:
    """
    너무 가까운 onset 병합.
    string buzz, 잔향 과감지 방지.
    음을 무시하는 게 아니라 앞 onset으로 합치는 것.

    예:
      [0.0, 0.05, 1.2, 1.22, 2.5] → [0.0, 1.2, 2.5]
      0.05는 0.0과 50ms 간격 → 과감지로 판단, 0.0 onset 유지
      1.22는 1.2와 20ms 간격 → 과감지로 판단, 1.2 onset 유지
    """
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
    """
    onset 기준으로 세그먼트 추출.
    각 onset에서 segment_duration(초)만큼 잘라냄.
    오디오 끝을 넘어가는 세그먼트는 제로 패딩.
    """
    segment_len = int(segment_duration * sr)
    segments = []

    for t in onset_times:
        start = int(t * sr)
        end = start + segment_len

        if end <= len(audio):
            seg = audio[start:end]
        else:
            # 오디오 끝 초과 → 제로 패딩
            seg = np.zeros(segment_len, dtype=np.float32)
            available = audio[start:]
            seg[:len(available)] = available

        # 묵음 세그먼트 필터링 (onset 감지됐지만 실제로 소리 없는 경우)
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
    """
    세그먼트를 output_dir/label/ 하위에 저장.
    기존 파일이 있으면 이어서 번호 매김 (덮어쓰기 방지).
    반환값: 저장된 세그먼트 수
    """
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # 기존 파일 수 확인 (이어서 번호 매김)
    existing = [
        f for f in os.listdir(label_dir)
        if f.endswith('.wav')
    ]
    start_idx = len(existing)

    for i, seg in enumerate(segments):
        filename = f"{label}_{start_idx + i:04d}.wav"
        filepath = os.path.join(label_dir, filename)
        sf.write(filepath, seg, sr)

    return len(segments)


def summarize(output_dir: str) -> None:
    """현재 수집된 샘플 수 출력"""
    print("\n── 현재 수집 현황 ──")
    for label in ['slap', 'pop', 'finger']:
        label_dir = os.path.join(output_dir, label)
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith('.wav')])
            status = "✅" if count >= 100 else f"({count}/100)"
            print(f"  {label:8s}: {count:3d}개 {status}")
        else:
            print(f"  {label:8s}:   0개 (0/100)")
    print()


def main():
    parser = argparse.ArgumentParser(description="베이스 주법 데이터 수집")
    parser.add_argument('--input', required=True, help="입력 wav 파일 경로")
    parser.add_argument('--label', required=True, choices=['slap', 'pop', 'finger'],
                        help="주법 레이블")
    parser.add_argument('--output_dir', default='./data/raw',
                        help="세그먼트 저장 디렉토리 (기본: data_collection/bass/data/raw)")
    parser.add_argument('--segment_duration', type=float, default=SEGMENT_DURATION,
                        help=f"세그먼트 길이 초 (기본: {SEGMENT_DURATION})")
    parser.add_argument('--min_interval', type=float, default=MIN_ONSET_INTERVAL,
                        help=f"onset 최소 간격 초 (기본: {MIN_ONSET_INTERVAL})")
    args = parser.parse_args()

    print(f"입력: {args.input}")
    print(f"레이블: {args.label}")

    # 1. 오디오 로드
    audio, sr = load_audio(args.input)
    print(f"오디오 로드 완료: {len(audio) / sr:.1f}초")

    # 2. onset 감지
    onset_times = detect_onsets(audio, sr)
    print(f"onset 감지: {len(onset_times)}개")

    # 3. onset 필터링
    onset_times = filter_onsets(onset_times, args.min_interval)
    print(f"필터링 후: {len(onset_times)}개")

    # 4. 세그먼트 추출
    segments = extract_segments(audio, sr, onset_times, args.segment_duration)
    print(f"유효 세그먼트: {len(segments)}개 (묵음 제거 후)")

    # 5. 저장
    saved = save_segments(segments, sr, args.label, args.output_dir)
    print(f"저장 완료: {args.output_dir}/{args.label}/")

    # 6. 현황 출력
    summarize(args.output_dir)


if __name__ == '__main__':
    main()