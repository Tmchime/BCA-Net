"""模型、预处理和训练参数。"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

VERSION = "2.3.0"


@dataclass
class Config:
    data_dir: str = "**"
    output_dir: str = "**"
    seeds: tuple = (6, 7, 42)
    split_seed: int = 42
    image_size: int = 224
    nir_dim: int = 499
    sg_window: int = 9
    sg_order: int = 5
    batch_size: int = 6
    eval_batch_size: int = 256
    workers: int = 4
    embed_dim: int = 512
    backbone: str = "convnext_base"
    pretrained: bool = True
    teacher_epochs: int = 40
    student_epochs: int = 60
    patience: int = 15
    lr: float = 4e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    min_lr: float = 1e-6
    mixup_alpha: float = 0.2
    temperature: float = 4.0
    ce_weight: float = 0.4
    kl_weight: float = 0.4
    attention_weight: float = 0.2
    coral_weight: float = 0.5
    grad_clip: float = 5.0
    tf32: bool = True
    device: str = "auto"
    strict_paper_counts: bool = True
    adapt_epochs: int = 15
    adapt_lr: float = 1e-4
    adapt_weight_decay: float = 1e-4
    loco_epochs: int | None = 60
    loco_lr: float | None = 4e-4
    loco_weight_decay: float | None = 0.05
    voc_columns: tuple = ("beta_caryophyllene", "camphene", "gamma_elemene", "propyl_2_butenoate")

    def dict(self):
        return json.loads(json.dumps(asdict(self)))


def load_config(path=None):
    values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = Config(**values)
    root = Path(__file__).resolve().parents[1]
    for attr in ["data_dir", "output_dir"]:
        if getattr(cfg, attr).strip() in {"", "**"}:
            raise ValueError(f"Replace '**' in {attr} with a local directory")
        value = Path(getattr(cfg, attr))
        setattr(cfg, attr, str((root / value).resolve()) if not value.is_absolute() else str(value))
    if cfg.batch_size < 2 or cfg.eval_batch_size < 1:
        raise ValueError("Training batch must be >=2 for BatchNorm/CORAL")
    if not 0 <= cfg.mixup_alpha:
        raise ValueError("mixup_alpha must be nonnegative")
    return cfg
