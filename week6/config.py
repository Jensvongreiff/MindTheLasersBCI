from dataclasses import dataclass


@dataclass(frozen=True)
class EEGNetConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    mixup: bool = False
    coral: bool = False
    lambda_coral: float = 0.0


EEGNET = EEGNetConfig(
    epochs=50,
    batch_size=16,
    lr=1e-3,
    mixup=False,
    coral=False,
    lambda_coral=0.0,
)


EEGNET_CORAL = EEGNetConfig(
    epochs=50,
    batch_size=64,
    lr=1e-3,
    mixup=False,
    coral=True,
    lambda_coral=0.05,
)


EEGNET_MIXUP = EEGNetConfig(
    epochs=50,
    batch_size=16,
    lr=1e-3,
    mixup=True,
    coral=False,
    lambda_coral=0.0,
)