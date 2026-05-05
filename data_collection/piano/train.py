"""
피아노 주법 분류 모델 학습 스크립트
inst-ai/data_collection/piano/train.py

사용법:
  python train.py --model crnn
  python train.py --model panns

사전 조건:
  preprocess_piano_data.py 실행 후
  crnn:  data/processed/X.npy, y.npy
  panns: data/processed/X_raw.npy, y.npy
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split

from model import get_model

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = './data/processed'
CHECKPOINT_DIR = './data/checkpoints'
PLOTS_DIR = './data/plots'

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
N_CLASSES = 3
LABELS = ['trill', 'arpeggio', 'normal']


class PianoDataset(Dataset):
    """
    crnn:  X (N, 128, T) → 채널 추가 → (N, 1, 128, T)
    panns: X (N, T)      → 그대로
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 model_type: str = 'crnn'):
        if model_type == 'panns':
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = torch.tensor(X[:, np.newaxis, :, :],
                                  dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_dataset(model_type: str) -> tuple[Dataset, Dataset, np.ndarray]:
    """
    반환값: (train_dataset, val_dataset, y_train)
    y_train: class weight 계산에 사용
    """
    y = np.load(os.path.join(PROCESSED_DIR, 'y.npy'))

    if model_type == 'panns':
        x_path = os.path.join(PROCESSED_DIR, 'X_raw.npy')
        if not os.path.exists(x_path):
            raise FileNotFoundError(
                "X_raw.npy 없음. preprocess_piano_data.py 다시 실행하세요.")
        X = np.load(x_path)
    else:
        X = np.load(os.path.join(PROCESSED_DIR, 'X.npy'))

    print(f"전체 데이터: {len(y)}개")
    for label, idx in zip(LABELS, range(N_CLASSES)):
        print(f"  {label}: {int(np.sum(y == idx))}개")

    dataset = PianoDataset(X, y, model_type=model_type)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator)

    # class weight 계산을 위해 학습 데이터의 y만 추출
    train_indices = train_dataset.indices
    y_train = y[train_indices]

    print(f"학습: {train_size}개 / 검증: {val_size}개")
    return train_dataset, val_dataset, y_train


def compute_class_weights(y_train: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """
    클래스별 샘플 수 기반 weight 계산.

    샘플이 적은 클래스일수록 높은 weight를 부여.
    틀렸을 때 더 크게 패널티를 줘서 불균형 보정.

    예:
      trill:373, arpeggio:305, normal:689
      weight ∝ 1/count → normal이 적은 weight
      → 모델이 normal만 예측하는 편향 방지
    """
    counts = np.array([
        int(np.sum(y_train == i)) for i in range(N_CLASSES)
    ], dtype=np.float32)

    weights = 1.0 / counts
    weights = weights / weights.sum()  # 합이 1이 되도록 정규화

    print("class weights:")
    for label, w, c in zip(LABELS, weights, counts):
        print(f"  {label}: {w:.4f} (샘플 {int(c)}개)")

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
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


def save_checkpoint(model: nn.Module, model_type: str, epoch: int,
                    val_loss: float) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{model_type}_best.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, path)


def plot_training_curve(
    train_losses: list, val_losses: list,
    train_accs: list, val_accs: list,
    model_type: str,
) -> None:

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


def main():
    parser = argparse.ArgumentParser(description="피아노 주법 분류 모델 학습")
    parser.add_argument('--model', required=True, choices=['crnn', 'panns'])
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

    train_dataset, val_dataset, y_train = load_dataset(args.model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)

    model = get_model(args.model).to(device)

    # class weight 적용
    class_weights = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY,
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
    plot_training_curve(train_losses, val_losses, train_accs, val_accs,
                        args.model)


if __name__ == '__main__':
    main()