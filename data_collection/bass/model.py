"""
베이스 주법 분류 모델
inst-ai/data_collection/bass/model.py

두 가지 모델 정의:
  - BassCNN: 처음부터 학습하는 2D-CNN (baseline)
  - BassFineTuned: PANNs CNN14 기반 Fine-tuning 모델

입력 형태: (batch, 1, 128, T)
  - batch: 배치 크기
  - 1: 채널 수 (흑백 이미지처럼 채널 1개)
  - 128: Mel 주파수 대역 수 (N_MELS)
  - T: 시간 프레임 수 (preprocess에서 고정됨)

출력 형태: (batch, 3)
  - 3: 클래스 수 (slap=0, pop=1, finger=2)
  - 각 값은 해당 클래스일 확률 (softmax 전 logit)
"""

import torch
import torch.nn as nn


# ── BassCNN (2D-CNN baseline) ─────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2D → BatchNorm → ReLU → MaxPool 묶음.

    반복되는 구조를 클래스로 분리한 이유:
    레이어 수를 바꾸거나 구조를 수정할 때 한 곳만 바꾸면 되기 때문.
    유지보수 원칙상 중복 코드는 묶는 게 맞음.

    in_channels: 입력 채널 수
    out_channels: 출력 채널 수 (필터 수). 많을수록 더 많은 패턴을 학습.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,   # 3x3 필터. 너무 크면 세부 패턴을 놓침
                padding=1,       # 입력과 출력 크기를 동일하게 유지
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 2x2 영역에서 최댓값만 남김 → 크기 절반
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BassCNN(nn.Module):
    """
    처음부터 학습하는 2D-CNN.

    구조:
      ConvBlock(1→32) → ConvBlock(32→64) → ConvBlock(64→128)
      → AdaptiveAvgPool → Flatten → Dropout → Linear(3)

    채널 수를 32→64→128로 늘리는 이유:
    앞 레이어는 단순한 패턴(엣지, 에너지 변화)을 잡고
    뒤 레이어는 더 복잡한 패턴(슬랩 attack 특성)을 잡음.
    복잡한 패턴일수록 더 많은 필터가 필요함.

    AdaptiveAvgPool을 쓰는 이유:
    입력 시간 길이(T)가 달라져도 동일한 크기로 출력.
    preprocess에서 고정했지만 혹시 모를 길이 차이에 대한 방어.
    """
    def __init__(self, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()

        self.conv_layers = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # Conv 후 공간 크기를 (4, 4)로 고정
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),   # 학습 중 50% 뉴런 랜덤 비활성화 → 과적합 방지
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 128, T)

        AdaptiveAvgPool2d가 MPS에서 입력 크기가 출력 크기로
        나눠지지 않으면 동작하지 않는 버그가 있음 (PyTorch #96056).
        pool 직전에 CPU로 이동 후 다시 원래 디바이스로 복귀.
        """
        device = x.device
        x = self.conv_layers(x)  # (batch, 128, H', W')
        x = self.pool(x.cpu()).to(device)  # MPS 버그 우회
        x = self.classifier(x)  # (batch, n_classes)
        return x


# ── BassFineTuned (PANNs CNN14 Fine-tuning) ───────────────

class BassFineTuned(nn.Module):
    """
    PANNs CNN14 기반 Fine-tuning 모델.

    PANNs (Pretrained Audio Neural Networks):
    Google AudioSet (YouTube 527개 오디오 클래스)으로 사전학습된 모델.
    오디오 Mel-spectrogram 특징 추출에 특화됨.

    Fine-tuning 전략 (Partial Fine-tuning):
    - 앞쪽 레이어 (conv_block0~3): 고정 (일반적인 오디오 특징)
    - 뒤쪽 레이어 (conv_block4~5): 학습 (고수준 특징 조정)
    - 분류 레이어: 새로 교체하여 학습

    앞쪽을 고정하는 이유:
    저수준 오디오 특징(onset, 에너지 변화 등)은 베이스에도 그대로 적용됨.
    처음부터 다시 학습하는 건 낭비이고 데이터가 적을 때 오히려 성능 저하.
    """
    def __init__(self, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()

        # PANNs CNN14 로드
        # pytorch_model_2020_Cnn14_mAP=0.431.pth 를 사용
        # 다운로드: https://zenodo.org/record/3987831
        self.cnn14 = self._load_cnn14()

        # 원래 분류 레이어 교체 (527 → n_classes)
        in_features = self.cnn14.fc_audioset.in_features
        self.cnn14.fc_audioset = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

        # Partial Fine-tuning: 앞쪽 4블록 고정
        self._freeze_front_layers()

    def _load_cnn14(self):
        """
        PANNs CNN14 모델 로드.
        panns_inference 패키지 사용.
        """
        try:
            from panns_inference import AudioTagging
            # panns_inference는 내부적으로 CNN14 구조를 포함함
            # 여기서는 구조만 가져오고 가중치는 별도 로드
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
            # 사전학습 가중치 로드 (파일 없으면 경고만 출력)
            import os
            weights_path = './panns_cnn14.pth'
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'], strict=False)
                print("PANNs CNN14 사전학습 가중치 로드 완료")
            else:
                print(f"⚠️  {weights_path} 없음. 가중치 없이 진행.")
                print("   다운로드: https://zenodo.org/record/3987831")
            return model
        except ImportError:
            raise ImportError(
                "panns_inference 패키지 필요: pip install panns_inference"
            )

    def _freeze_front_layers(self):
        """
        앞쪽 4개 Conv 블록 고정.
        requires_grad=False: 해당 레이어는 역전파 시 가중치 업데이트 안 함.
        """
        freeze_layers = [
            'conv_block0', 'conv_block1',
            'conv_block2', 'conv_block3',
        ]
        for name, param in self.cnn14.named_parameters():
            if any(name.startswith(layer) for layer in freeze_layers):
                param.requires_grad = False

        frozen = sum(
            p.numel() for p in self.cnn14.parameters() if not p.requires_grad
        )
        total = sum(p.numel() for p in self.cnn14.parameters())
        print(f"고정된 파라미터: {frozen:,} / 전체: {total:,} "
              f"({frozen/total*100:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T) ← raw audio
        CNN14는 (batch, T) 형태의 raw audio를 직접 입력으로 받음.
        내부에서 Mel-spectrogram 변환을 직접 처리함.
        """
        output_dict = self.cnn14(x)
        logits = output_dict['clipwise_output']  # (batch, n_classes)
        return logits


# ── 모델 선택 헬퍼 ────────────────────────────────────────

def get_model(model_type: str, n_classes: int = 3) -> nn.Module:
    """
    model_type: 'cnn' 또는 'panns'
    """
    if model_type == 'cnn':
        model = BassCNN(n_classes=n_classes)
        total = sum(p.numel() for p in model.parameters())
        print(f"BassCNN 파라미터 수: {total:,}")
        return model
    elif model_type == 'panns':
        model = BassFineTuned(n_classes=n_classes)
        return model
    else:
        raise ValueError(f"model_type은 'cnn' 또는 'panns'만 가능: {model_type}")


if __name__ == '__main__':
    # 구조 확인용 테스트
    # preprocess 기준: N_MELS=128, SEGMENT_DURATION=0.5, HOP_LENGTH=512, SR=44100
    # T = 0.5 * 44100 / 512 = 43 프레임
    dummy_input = torch.randn(4, 1, 128, 43)  # batch=4

    print("=== BassCNN ===")
    cnn = BassCNN()
    out = cnn(dummy_input)
    print(f"입력: {dummy_input.shape} → 출력: {out.shape}")  # (4, 3)
    assert out.shape == (4, 3)
    print("구조 확인 완료\n")