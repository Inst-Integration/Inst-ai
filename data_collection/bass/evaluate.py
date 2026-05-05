"""
베이스 주법 분류 모델 평가 스크립트
inst-ai/data_collection/bass/evaluate.py

사용법:
  python evaluate.py --model cnn
  python evaluate.py --model panns
  python evaluate.py --model both   ← 둘 다 비교

사전 조건:
  train.py 실행 후 data/checkpoints/ 아래 .pth 파일 있어야 함

출력:
  콘솔: accuracy, precision, recall, f1 (클래스별 + 전체)
  data/plots/
    cnn_confusion_matrix.png
    panns_confusion_matrix.png
    comparison_table.png   ← --model both 일 때만
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader, Dataset, random_split

from model import get_model

# ── 파라미터 (train.py와 동일하게 맞춰야 함) ──────────────
PROCESSED_DIR = './data/processed'
CHECKPOINT_DIR = './data/checkpoints'
PLOTS_DIR = './data/plots'
BATCH_SIZE = 16
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
LABELS = ['slap', 'pop', 'finger']


# ── 데이터셋 ──────────────────────────────────────────────

class BassDataset(Dataset):
    """
    cnn:   X (N, 128, T) → 채널 추가 → (N, 1, 128, T)
    panns: X (N, T)      → 그대로
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, model_type: str = 'cnn'):
        if model_type == 'panns':
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_val_dataset(model_type: str = 'cnn') -> DataLoader:
    """
    train.py와 동일한 seed로 분리해서 검증 데이터만 가져옴.
    seed가 다르면 학습 데이터가 검증에 섞여서 정확도가 부풀려짐.
    model_type에 따라 X.npy 또는 X_raw.npy 로드.
    """
    if model_type == 'panns':
        x_path = os.path.join(PROCESSED_DIR, 'X_raw.npy')
    else:
        x_path = os.path.join(PROCESSED_DIR, 'X.npy')

    X = np.load(x_path)
    y = np.load(os.path.join(PROCESSED_DIR, 'y.npy'))

    dataset = BassDataset(X, y, model_type=model_type)
    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator)

    return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ── 평가 ──────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    전체 검증 데이터에 대해 예측 실행.
    반환: (실제 레이블 배열, 예측 레이블 배열)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Accuracy, Precision, Recall, F1 계산.
    sklearn 없이 numpy로 직접 계산.

    Precision: 슬랩이라고 예측한 것 중 실제 슬랩 비율
    Recall:    실제 슬랩 중 슬랩이라고 예측한 비율
    F1:        Precision과 Recall의 조화평균

    macro average: 클래스별 점수를 단순 평균
    """
    n_classes = len(LABELS)
    metrics = {}

    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        metrics[LABELS[c]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(y_true == c)),
        }

    metrics['accuracy'] = float(np.mean(y_true == y_pred))
    metrics['macro_precision'] = float(np.mean(precisions))
    metrics['macro_recall'] = float(np.mean(recalls))
    metrics['macro_f1'] = float(np.mean(f1s))

    return metrics


def print_metrics(metrics: dict, model_type: str) -> None:
    print(f"\n=== {model_type} 평가 결과 ===")
    print(f"{'':10s} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    print("-" * 50)
    for label in LABELS:
        m = metrics[label]
        print(f"{label:10s} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10d}")
    print("-" * 50)
    print(f"{'accuracy':10s} {'':>10} {'':>10} "
          f"{metrics['accuracy']:>10.3f} {sum(metrics[l]['support'] for l in LABELS):>10d}")
    print(f"{'macro avg':10s} {metrics['macro_precision']:>10.3f} "
          f"{metrics['macro_recall']:>10.3f} {metrics['macro_f1']:>10.3f}")


# ── 혼동 행렬 시각화 ──────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
) -> None:
    """
    혼동 행렬 (Confusion Matrix).
    행: 실제 레이블 / 열: 예측 레이블
    대각선: 맞게 예측한 것
    대각선 외: 틀린 것 (어떤 클래스를 어떤 클래스로 헷갈리는지 파악)
    """

    os.makedirs(PLOTS_DIR, exist_ok=True)

    n = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_type} - Confusion Matrix')

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]),
                    ha='center', va='center',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'{model_type}_confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"혼동 행렬 저장: {out_path}")


def plot_comparison_table(results: dict) -> None:
    """cnn vs panns 비교표 시각화."""

    os.makedirs(PLOTS_DIR, exist_ok=True)

    models = list(results.keys())
    metrics_keys = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    table_data = []
    for m_key, m_label in zip(metrics_keys, metric_labels):
        row = [m_label]
        for model_type in models:
            row.append(f"{results[model_type][m_key]:.3f}")
        table_data.append(row)

    col_labels = ['Metric'] + models
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    ax.set_title('2D-CNN vs PANNs Fine-tuning', pad=20)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'comparison_table.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"비교표 저장: {out_path}")


# ── 단일 모델 평가 ────────────────────────────────────────

def evaluate_model(model_type: str, val_loader: DataLoader,
                   device: torch.device) -> dict:
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_type}_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  {checkpoint_path} 없음. train.py 먼저 실행하세요.")
        return {}

    model = get_model(model_type).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"{model_type} 체크포인트 로드 (epoch {checkpoint['epoch']}, "
          f"val_loss {checkpoint['val_loss']:.4f})")

    y_true, y_pred = run_inference(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, model_type)
    plot_confusion_matrix(y_true, y_pred, model_type)

    return metrics


# ── 메인 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="베이스 주법 분류 모델 평가")
    parser.add_argument('--model', required=True,
                        choices=['cnn', 'panns', 'both'])
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.model == 'both':
        results = {}
        for model_type in ['cnn', 'panns']:
            val_loader = load_val_dataset(model_type)
            metrics = evaluate_model(model_type, val_loader, device)
            if metrics:
                results[model_type] = metrics
        if len(results) == 2:
            plot_comparison_table(results)
    else:
        val_loader = load_val_dataset(args.model)
        evaluate_model(args.model, val_loader, device)


if __name__ == '__main__':
    main()