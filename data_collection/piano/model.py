"""
피아노 주법 분류 모델
inst-ai/data_collection/piano/model.py

모델:
  PianoCRNN: CRNN (CNN + GRU) baseline
  PianoFineTuned: PANNs CNN14 기반 Fine-tuning

입력 형태: (batch, 1, 128, T)
  - T: 2초 세그먼트 기준 프레임 수 (~172프레임)

출력 형태: (batch, 3)
  - 0=trill, 1=arpeggio, 2=normal
"""

import os

import torch
import torch.nn as nn


# ── PianoCRNN ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2D → BatchNorm → ReLU → MaxPool 블록.

    피아노는 베이스보다 주파수 범위가 넓어서
    MaxPool을 주파수 축(height)으로만 적용.
    시간 축은 GRU가 처리하기 때문에 보존.

    pool_size: (주파수 축 풀링, 시간 축 풀링)
    시간 축을 1로 유지하면 프레임 수가 줄지 않음.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 pool_size: tuple = (2, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PianoCRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) baseline.

    구조:
      CNN 부분: 주파수 특성 추출
        ConvBlock(1→32)   → (batch, 32, 64, T)   주파수 64로 압축
        ConvBlock(32→64)  → (batch, 64, 32, T)   주파수 32로 압축
        ConvBlock(64→128) → (batch, 128, 16, T)  주파수 16으로 압축

      reshape: (batch, 128*16, T) → (batch, T, 128*16)
        GRU는 (batch, 시퀀스길이, 특징수) 형태를 기대함

      GRU 부분: 시간 패턴 학습
        GRU(input=128*16, hidden=256, layers=2)
        트릴/아르페지오의 onset 간격 패턴을 시간 순서로 학습

      FC: 최종 분류
        마지막 시간 스텝의 hidden state → 3클래스

    왜 시간 축 MaxPool을 안 하냐:
      트릴/아르페지오 감지는 onset 간격 패턴이 핵심.
      시간 축을 풀링하면 프레임이 줄어서 패턴이 뭉개질 수 있음.
      GRU가 전체 시간 축을 처리하도록 보존.
    """
    def __init__(self, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()

        self.conv_layers = nn.Sequential(
            ConvBlock(1, 32, pool_size=(2, 1)),
            ConvBlock(32, 64, pool_size=(2, 1)),
            ConvBlock(64, 128, pool_size=(2, 1)),
        )

        # CNN 출력: (batch, 128, 16, T)
        # GRU 입력 특징 수: 128 * 16 = 2048
        self.gru_input_size = 128 * 16

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=256,
            num_layers=2,        # 2층 GRU: 더 복잡한 시간 패턴 학습 가능
            batch_first=True,    # (batch, T, features) 형태로 입력
            dropout=dropout,     # GRU 레이어 간 dropout
            bidirectional=False, # 단방향: 실시간 처리 고려
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 128, T)
        """
        # CNN: 주파수 특성 추출
        x = self.conv_layers(x)          # (batch, 128, 16, T)

        batch, channels, freq, time = x.shape

        # reshape: 주파수와 채널을 합쳐서 GRU 입력 형태로
        x = x.permute(0, 3, 1, 2)        # (batch, T, 128, 16)
        x = x.reshape(batch, time, -1)   # (batch, T, 128*16)

        # GRU: 시간 패턴 학습
        x, _ = self.gru(x)               # (batch, T, 256)

        # 마지막 시간 스텝만 사용
        # 트릴/아르페지오 패턴은 시퀀스 전체를 본 후 판단
        x = x[:, -1, :]                  # (batch, 256)

        x = self.classifier(x)           # (batch, n_classes)
        return x


# ── PianoFineTuned (PANNs CNN14) ──────────────────────────

class PianoFineTuned(nn.Module):
    """
    PANNs CNN14 기반 Fine-tuning.
    베이스와 동일한 구조, 클래스만 다름 (trill/arpeggio/normal).
    샘플레이트 불일치 이슈는 베이스와 동일하게 존재.
    """
    def __init__(self, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.cnn14 = self._load_cnn14()

        in_features = self.cnn14.fc_audioset.in_features
        self.cnn14.fc_audioset = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )
        self._freeze_front_layers()

    def _load_cnn14(self):
        try:
            import panns_inference.models as panns_models
            model = panns_models.Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            weights_path = './panns_cnn14.pth'
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'], strict=False)
                print("PANNs CNN14 사전학습 가중치 로드 완료")
            else:
                print(f"⚠️  {weights_path} 없음.")
            return model
        except ImportError:
            raise ImportError("pip install panns_inference")

    def _freeze_front_layers(self):
        freeze_layers = ['conv_block0', 'conv_block1',
                         'conv_block2', 'conv_block3']
        for name, param in self.cnn14.named_parameters():
            if any(name.startswith(l) for l in freeze_layers):
                param.requires_grad = False

        frozen = sum(p.numel() for p in self.cnn14.parameters()
                     if not p.requires_grad)
        total = sum(p.numel() for p in self.cnn14.parameters())
        print(f"고정된 파라미터: {frozen:,} / 전체: {total:,} "
              f"({frozen/total*100:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T) ← raw audio"""
        output_dict = self.cnn14(x)
        return output_dict['clipwise_output']


# ── 모델 선택 헬퍼 ────────────────────────────────────────

def get_model(model_type: str, n_classes: int = 3) -> nn.Module:
    """model_type: 'crnn' 또는 'panns'"""
    if model_type == 'crnn':
        model = PianoCRNN(n_classes=n_classes)
        total = sum(p.numel() for p in model.parameters())
        print(f"PianoCRNN 파라미터 수: {total:,}")
        return model
    elif model_type == 'panns':
        return PianoFineTuned(n_classes=n_classes)
    else:
        raise ValueError(f"model_type은 'crnn' 또는 'panns'만 가능: {model_type}")


if __name__ == '__main__':
    # 구조 확인용 테스트
    # N_MELS=128, SEGMENT_DURATION=2.0, HOP_LENGTH=512, SR=44100
    # T = 2.0 * 44100 / 512 = 172 프레임
    dummy_input = torch.randn(4, 1, 128, 172)  # batch=4

    print("=== PianoCRNN ===")
    crnn = PianoCRNN()
    out = crnn(dummy_input)
    print(f"입력: {dummy_input.shape} → 출력: {out.shape}")
    assert out.shape == (4, 3)
    print("구조 확인 완료")