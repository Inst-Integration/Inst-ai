"""
베이스 주법 분류 모델 학습 스크립트
inst-ai/data_collection/bass/train.py

사용법:
  python train.py --model cnn      ← BassCNN (2D-CNN baseline)
  python train.py --model panns    ← PANNs Fine-tuning

사전 조건:
  preprocess_bass_data.py 실행 후
  cnn:   data/processed/X.npy, y.npy
  panns: data/processed/X_raw.npy, y.npy

출력:
  data/checkpoints/
    cnn_best.pth
    panns_best.pth
  data/plots/
    cnn_training_curve.png
    panns_training_curve.png
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

# ── 파라미터 ──────────────────────────────────────────────
PROCESSED_DIR = './data/processed'
CHECKPOINT_DIR = './data/checkpoints'
PLOTS_DIR = './data/plots'

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ── 데이터셋 ──────────────────────────────────────────────

class BassDataset(Dataset):
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


def load_dataset(model_type: str) -> tuple[Dataset, Dataset]:
    """
    model_type에 따라 다른 X 파일 로드.
    cnn   → X.npy     (Mel-spectrogram)
    panns → X_raw.npy (raw audio)
    y.npy는 공통.
    """
    y = np.load(os.path.join(PROCESSED_DIR, 'y.npy'))

    if model_type == 'panns':
        x_path = os.path.join(PROCESSED_DIR, 'X_raw.npy')
        if not os.path.exists(x_path):
            raise FileNotFoundError(
                "X_raw.npy 없음. preprocess_bass_data.py 다시 실행하세요."
            )
        X = np.load(x_path)
    else:
        X = np.load(os.path.join(PROCESSED_DIR, 'X.npy'))

    print(f"전체 데이터: {len(y)}개")
    for label, idx in {'slap': 0, 'pop': 1, 'finger': 2}.items():
        print(f"  {label}: {int(np.sum(y == idx))}개")

    dataset = BassDataset(X, y, model_type=model_type)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator)

    print(f"학습: {train_size}개 / 검증: {val_size}개")
    return train_dataset, val_dataset


# ── 학습 루프 ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    에폭 하나 학습.
    반환: (평균 손실, 정확도)

    model.train(): Dropout 활성화, BatchNorm 학습 모드
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / len(loader), correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    검증 데이터로 평가.
    반환: (평균 손실, 정확도)

    model.eval(): Dropout 비활성화, BatchNorm 추론 모드
    torch.no_grad(): gradient 계산 안 함 → 메모리/속도 절약
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / len(loader), correct / total


# ── 체크포인트 ────────────────────────────────────────────

def save_checkpoint(model: nn.Module, model_type: str, epoch: int,
                    val_loss: float) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{model_type}_best.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, path)


# ── 학습 곡선 시각화 ──────────────────────────────────────

def plot_training_curve(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    model_type: str,
) -> None:
    """
    학습/검증 손실 및 정확도 곡선.
    학습 정확도는 올라가는데 검증 정확도가 내려가면 과적합.
    """

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='train loss')
    ax1.plot(epochs, val_losses, label='val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_type} - Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, label='train acc')
    ax2.plot(epochs, val_accs, label='val acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_type} - Accuracy')
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'{model_type}_training_curve.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"학습 곡선 저장: {out_path}")
    print("→ val acc가 train acc보다 크게 낮으면 과적합 의심")


# ── 메인 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="베이스 주법 분류 모델 학습")
    parser.add_argument('--model', required=True, choices=['cnn', 'panns'],
                        help="모델 선택: cnn (2D-CNN baseline) / panns (Fine-tuning)")
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"디바이스: {device}")

    train_dataset, val_dataset = load_dataset(args.model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)

    model = get_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n=== {args.model} 학습 시작 ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} | "
                  f"val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, args.model, epoch, val_loss)

    print(f"\n학습 완료. 최적 val_loss: {best_val_loss:.4f}")
    print(f"체크포인트: {CHECKPOINT_DIR}/{args.model}_best.pth")

    plot_training_curve(train_losses, val_losses, train_accs, val_accs,
                        args.model)


if __name__ == '__main__':
    main()