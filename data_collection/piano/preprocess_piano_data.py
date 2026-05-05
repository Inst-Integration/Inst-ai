"""
피아노 주법 데이터 전처리 및 시각화 스크립트
inst-ai/data_collection/piano/preprocess_piano_data.py

베이스와 다른 점:
  - SEGMENT_DURATION = 2.0
  - LABELS = ['trill', 'arpeggio', 'normal']
  - ATK_R 분포 시각화 없음 (피아노는 rule-based ATK_R 안 씀)

출력:
  data/processed/
    X.npy        ← Mel-spectrogram (CRNN용)
    X_raw.npy    ← raw audio (PANNs용)
    y.npy
    label_map.json
  data/plots/
    mel_samples.png
    rms_dist.png
"""

import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

SR = 44100
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_DURATION = 2.0

RAW_DIR = './data/raw'
PROCESSED_DIR = './data/processed'
PLOTS_DIR = './data/plots'

LABELS = ['trill', 'arpeggio', 'normal']
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}


def load_segment(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    target_len = int(SEGMENT_DURATION * SR)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio


def to_melspectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def load_all_segments() -> tuple[dict, dict]:
    audios = {label: [] for label in LABELS}
    rms_vals = {label: [] for label in LABELS}

    for label in LABELS:
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.exists(label_dir):
            print(f"  ⚠️  {label_dir} 없음, 건너뜀")
            continue

        files = sorted([f for f in os.listdir(label_dir) if f.endswith('.wav')])
        if not files:
            print(f"  ⚠️  {label}: 파일 없음")
            continue

        for fname in files:
            audio = load_segment(os.path.join(label_dir, fname))
            audios[label].append(audio)
            rms_vals[label].append(compute_rms(audio))

        print(f"  {label}: {len(files)}개 로드")

    return audios, rms_vals


def plot_mel_samples(audios: dict, n_samples: int = 3) -> None:
    """
    주법별 Mel-spectrogram 샘플 시각화.

    트릴: 짧은 간격의 onset이 반복되는 패턴이 시간 축에서 보여야 함
    아르페지오: onset 간격이 트릴보다 넓고 음정 변화가 보여야 함
    일반연주: 위 두 패턴과 구분되는 형태
    패턴 차이가 육안으로 보이는지 확인하세요.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    n_labels = len(LABELS)
    fig, axes = plt.subplots(n_labels, n_samples,
                             figsize=(n_samples * 4, n_labels * 3))
    fig.suptitle('Mel-spectrogram samples', fontsize=14)

    for row, label in enumerate(LABELS):
        samples = audios[label][:n_samples]
        for col, audio in enumerate(samples):
            ax = axes[row][col]
            mel_db = to_melspectrogram(audio)
            librosa.display.specshow(
                mel_db, sr=SR, hop_length=HOP_LENGTH,
                x_axis='time', y_axis='mel', ax=ax)
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(f'{label} #{col + 1}')

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'mel_samples.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  저장: {out_path}")
    print("  → 트릴: 짧은 간격 반복 패턴이 보이는지 확인")
    print("  → 아르페지오: 트릴보다 넓은 간격, 음정 변화가 보이는지 확인")


def plot_rms_dist(rms_vals: dict) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'trill': '#e74c3c', 'arpeggio': '#3498db', 'normal': '#2ecc71'}

    for label in LABELS:
        values = rms_vals[label]
        if not values:
            continue
        ax.hist(values, bins=30, alpha=0.6, label=label, color=colors[label])

    ax.set_xlabel('RMS')
    ax.set_ylabel('count')
    ax.set_title('RMS distribution - onset quality check')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'rms_dist.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  저장: {out_path}")
    print("  → RMS 0.001 미만 샘플 많으면 묵음 세그먼트 의심")


def build_dataset_mel(audios: dict) -> tuple[np.ndarray, np.ndarray]:
    """CRNN용 Mel-spectrogram 데이터셋"""
    all_mels = []
    all_labels = []

    dummy = np.zeros(int(SEGMENT_DURATION * SR))
    target_frames = to_melspectrogram(dummy).shape[1]

    for label in LABELS:
        for audio in audios[label]:
            mel = to_melspectrogram(audio)
            if mel.shape[1] < target_frames:
                mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])),
                             mode='constant', constant_values=mel.min())
            else:
                mel = mel[:, :target_frames]
            all_mels.append(mel)
            all_labels.append(LABEL_MAP[label])

    X = np.array(all_mels, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def build_dataset_raw(audios: dict) -> np.ndarray:
    """PANNs CNN14용 raw audio 데이터셋"""
    all_audios = []
    target_len = int(SEGMENT_DURATION * SR)

    for label in LABELS:
        for audio in audios[label]:
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            all_audios.append(audio)

    return np.array(all_audios, dtype=np.float32)


def save_dataset(X_mel: np.ndarray, X_raw: np.ndarray,
                 y: np.ndarray) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'X.npy'), X_mel)
    np.save(os.path.join(PROCESSED_DIR, 'X_raw.npy'), X_raw)
    np.save(os.path.join(PROCESSED_DIR, 'y.npy'), y)
    with open(os.path.join(PROCESSED_DIR, 'label_map.json'), 'w') as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    print(f"  X (mel): {X_mel.shape}  (샘플 수, 주파수 대역, 시간 프레임)")
    print(f"  X_raw:   {X_raw.shape}  (샘플 수, 시간 샘플)")
    print(f"  y:       {y.shape}")
    for label, idx in LABEL_MAP.items():
        print(f"    {label}: {int(np.sum(y == idx))}개")


def main():
    print("=== 1. 세그먼트 로드 ===")
    audios, rms_vals = load_all_segments()

    total = sum(len(v) for v in audios.values())
    if total == 0:
        print("데이터 없음. collect_piano_data.py 먼저 실행하세요.")
        return

    print(f"\n=== 2. 시각화 ===")
    plot_mel_samples(audios)
    plot_rms_dist(rms_vals)

    print(f"\n=== 3. 데이터셋 저장 ===")
    X_mel, y = build_dataset_mel(audios)
    X_raw = build_dataset_raw(audios)
    save_dataset(X_mel, X_raw, y)

    print(f"\n완료. data/plots/ 에서 시각화 결과 확인 후 다음 단계 진행하세요.")


if __name__ == '__main__':
    main()