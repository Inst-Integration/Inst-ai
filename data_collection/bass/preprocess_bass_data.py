"""
베이스 주법 데이터 전처리 및 시각화 스크립트
inst-ai/data_collection/preprocess_bass_data.py

사용법:
  python preprocess_bass_data.py

실행 전 조건:
  collect_bass_data.py로 data/raw/ 아래 세그먼트가 수집돼 있어야 함
  data/raw/
    slap/  ← slap_0000.wav ~ slap_00N.wav
    pop/   ← pop_0000.wav ~ pop_00N.wav
    finger/ ← finger_0000.wav ~ finger_00N.wav

출력:
  data/processed/
    X.npy        ← Mel-spectrogram 배열 (N x 128 x T)
    y.npy        ← 레이블 배열 (N,)  0=slap, 1=pop, 2=finger
    label_map.json
  data/plots/
    mel_samples.png    ← 주법별 Mel-spectrogram 샘플
    atk_r_dist.png     ← ATK_R 분포 (threshold 검증용)
    rms_dist.png       ← RMS 분포 (onset 감지 품질 확인)
"""

import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ── 파라미터 ──────────────────────────────────────────────
SR = 44100
N_MELS = 128          # Mel 주파수 대역 수. 64~256 범위에서 조정 가능
HOP_LENGTH = 512      # 프레임 간격 (샘플 수). 작을수록 시간 해상도 높아짐
N_FFT = 2048          # FFT 윈도우 크기. 주파수 해상도에 영향
SEGMENT_DURATION = 0.5

RAW_DIR = './data/raw'              # data_collection/bass/data/raw
PROCESSED_DIR = './data/processed'  # data_collection/bass/data/processed
PLOTS_DIR = './data/plots'          # data_collection/bass/data/plots

LABELS = ['slap', 'pop', 'finger']
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}


# ── 유틸 ──────────────────────────────────────────────────

def load_segment(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    # 길이 고정 (짧으면 패딩, 길면 자름)
    target_len = int(SEGMENT_DURATION * SR)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio


def to_melspectrogram(audio: np.ndarray) -> np.ndarray:
    """
    오디오 → Mel-spectrogram (dB 스케일)

    power_to_db: 에너지값을 dB로 변환. 사람 귀가 로그 스케일로 소리를 인식하기 때문에
    dB 변환하면 모델이 학습하기 더 좋은 표현이 됨.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_atk_r(audio: np.ndarray) -> float:
    """
    ATK_R 계산. dsp_bridge.c의 로직과 동일하게 맞춤.
    peak_rms / avg_rms 비율.

    onset 이후 짧은 구간(attack window)의 RMS와
    전체 구간 평균 RMS의 비율로 attack 강도를 측정.
    슬랩은 attack이 강해서 ATK_R이 높고,
    팝/핑거링은 상대적으로 낮음.
    """
    attack_window = int(0.02 * SR)  # 20ms attack window
    if len(audio) < attack_window:
        return 0.0

    attack_rms = np.sqrt(np.mean(audio[:attack_window] ** 2))
    avg_rms = np.sqrt(np.mean(audio ** 2))

    if avg_rms < 1e-8:
        return 0.0

    return float(attack_rms / avg_rms)


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


# ── 데이터 로드 ───────────────────────────────────────────

def load_all_segments() -> tuple[dict, dict, dict]:
    """
    반환값:
      audios:  {label: [audio, ...]}
      atk_rs:  {label: [atk_r, ...]}
      rms_vals: {label: [rms, ...]}
    """
    audios = {label: [] for label in LABELS}
    atk_rs = {label: [] for label in LABELS}
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
            path = os.path.join(label_dir, fname)
            audio = load_segment(path)
            audios[label].append(audio)
            atk_rs[label].append(compute_atk_r(audio))
            rms_vals[label].append(compute_rms(audio))

        print(f"  {label}: {len(files)}개 로드")

    return audios, atk_rs, rms_vals


# ── 시각화 ────────────────────────────────────────────────

def plot_mel_samples(audios: dict, n_samples: int = 3) -> None:
    """
    주법별 Mel-spectrogram 샘플 시각화.

    목적: 슬랩/팝/핑거링의 주파수 분포 패턴 차이가 육안으로 보이는지 확인.
    패턴 차이가 명확할수록 2D-CNN이 잘 구분할 수 있다는 근거가 됨.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    n_labels = len(LABELS)
    fig, axes = plt.subplots(n_labels, n_samples, figsize=(n_samples * 4, n_labels * 3))
    fig.suptitle('Mel-spectrogram 샘플 (주법별)', fontsize=14)

    for row, label in enumerate(LABELS):
        samples = audios[label][:n_samples]
        for col, audio in enumerate(samples):
            ax = axes[row][col]
            mel_db = to_melspectrogram(audio)
            librosa.display.specshow(
                mel_db,
                sr=SR,
                hop_length=HOP_LENGTH,
                x_axis='time',
                y_axis='mel',
                ax=ax,
            )
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(f'{label} #{col + 1}')

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'mel_samples.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  저장: {out_path}")
    print("  → 주법별 패턴 차이가 육안으로 보이는지 확인하세요")
    print("  → 슬랩: attack 순간 저주파 에너지 강할 것으로 예상")
    print("  → 팝: 고주파 성분이 슬랩보다 강할 것으로 예상")
    print("  → 실제 데이터에서 다를 수 있음 — 패턴 직접 확인 필요")


def plot_atk_r_dist(atk_rs: dict) -> None:
    """
    ATK_R 분포 히스토그램.

    목적: dsp_bridge.c의 threshold 2.3 적정성 검증.
    슬랩과 팝/핑거링의 ATK_R 분포가 명확히 분리되면 threshold가 맞는 것.
    겹치는 구간이 많으면 threshold 재조정 필요.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'slap': '#e74c3c', 'pop': '#3498db', 'finger': '#2ecc71'}

    for label in LABELS:
        values = atk_rs[label]
        if not values:
            continue
        ax.hist(values, bins=30, alpha=0.6, label=label, color=colors[label])

    # 현재 threshold 표시
    ax.axvline(x=2.3, color='black', linestyle='--', linewidth=1.5,
               label='현재 threshold (2.3)')

    ax.set_xlabel('ATK_R (peak_rms / avg_rms)')
    ax.set_ylabel('샘플 수')
    ax.set_title('ATK_R 분포 — threshold 검증용')
    ax.legend()

    # 통계 출력
    stats_text = []
    for label in LABELS:
        values = atk_rs[label]
        if values:
            stats_text.append(
                f"{label}: mean={np.mean(values):.2f}, "
                f"std={np.std(values):.2f}, "
                f"min={np.min(values):.2f}, "
                f"max={np.max(values):.2f}"
            )
    ax.text(0.02, 0.98, '\n'.join(stats_text),
            transform=ax.transAxes, verticalalignment='top',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'atk_r_dist.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  저장: {out_path}")
    print("  → 슬랩 분포와 팝/핑거링 분포가 2.3 기준으로 분리되는지 확인하세요")
    print("  → 겹치는 구간이 크면 threshold 재조정 또는 딥러닝 필요성 근거가 됨")


def plot_rms_dist(rms_vals: dict) -> None:
    """
    RMS 분포 히스토그램.

    목적: onset 감지 품질 확인.
    RMS가 너무 낮은 샘플이 많으면 묵음 세그먼트가 포함됐거나
    onset 감지가 잘못된 것. 수집 품질 점검용.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'slap': '#e74c3c', 'pop': '#3498db', 'finger': '#2ecc71'}

    for label in LABELS:
        values = rms_vals[label]
        if not values:
            continue
        ax.hist(values, bins=30, alpha=0.6, label=label, color=colors[label])

    ax.set_xlabel('RMS')
    ax.set_ylabel('샘플 수')
    ax.set_title('RMS 분포 — onset 감지 품질 확인용')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'rms_dist.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  저장: {out_path}")
    print("  → RMS가 0.001 미만인 샘플이 많으면 묵음 세그먼트 포함 의심")
    print("  → collect_bass_data.py의 TOP_DB 파라미터 조정 고려")


# ── 전처리 및 저장 ────────────────────────────────────────

# preprocess_bass_data.py 하단 build_dataset 함수를 이걸로 교체

def build_dataset_mel(audios: dict) -> tuple[np.ndarray, np.ndarray]:
    """BassCNN용 Mel-spectrogram 데이터셋 (기존)"""
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


def build_dataset_raw(audios: dict) -> tuple[np.ndarray, np.ndarray]:
    """PANNs CNN14용 raw audio 데이터셋"""
    all_audios = []
    all_labels = []

    target_len = int(SEGMENT_DURATION * SR)

    for label in LABELS:
        for audio in audios[label]:
            # 길이 고정
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            all_audios.append(audio)
            all_labels.append(LABEL_MAP[label])

    X = np.array(all_audios, dtype=np.float32)  # (N, T)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def save_dataset(X_mel: np.ndarray, X_raw: np.ndarray,
                 y: np.ndarray) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, 'X.npy'), X_mel)      # BassCNN용
    np.save(os.path.join(PROCESSED_DIR, 'X_raw.npy'), X_raw)  # PANNs용
    np.save(os.path.join(PROCESSED_DIR, 'y.npy'), y)

    with open(os.path.join(PROCESSED_DIR, 'label_map.json'), 'w') as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    print(f"  X (mel): {X_mel.shape}")
    print(f"  X_raw:   {X_raw.shape}")
    print(f"  y:       {y.shape}")
    for label, idx in LABEL_MAP.items():
        print(f"    {label}: {int(np.sum(y == idx))}개")


# ── 메인 ──────────────────────────────────────────────────

def main():
    print("=== 1. 세그먼트 로드 ===")
    audios, atk_rs, rms_vals = load_all_segments()

    total = sum(len(v) for v in audios.values())
    if total == 0:
        print("데이터 없음. collect_bass_data.py 먼저 실행하세요.")
        return

    print(f"\n=== 2. 시각화 ===")
    plot_mel_samples(audios)
    plot_atk_r_dist(atk_rs)
    plot_rms_dist(rms_vals)

    print(f"\n=== 3. 데이터셋 저장 ===")
    X_mel, y = build_dataset_mel(audios)
    X_raw, _ = build_dataset_raw(audios)
    save_dataset(X_mel, X_raw, y)

    print(f"\n완료. data/plots/ 에서 시각화 결과 확인 후 다음 단계 진행하세요.")


if __name__ == '__main__':
    main()