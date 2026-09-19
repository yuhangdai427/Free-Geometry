"""Strict, serializable configuration; no model or torch imports."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class SamplingConfig:
    teacher_frames: int = 8
    student_frames: int = 4
    train_tasks: int = 10
    probe_tasks: int = 2
    dense_min_frames: int = 50
    sift_threshold: float = 0.6
    sift_mode: str = "auto"
    dense_candidates: int = 40
    seed: int = 0


@dataclass
class ModelConfig:
    name: str = "da3"
    weights: str = ""
    source: str = ""
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    resolution: int = 0


@dataclass
class LossConfig:
    feature: float = 1.0
    rotation: float = 1.0
    translation: float = 1.0
    rkd: float = 1.0
    couple: float = 1.0
    rkd_delta: float = 0.2
    rotation_knee_deg: float = 20.0


@dataclass
class ReliabilityConfig:
    enabled: bool = True
    feature: bool = True
    rotation: bool = True
    translation: bool = True
    rkd: bool = True
    couple: bool = True
    feature_threshold: float = 0.1
    rotation_threshold_deg: float = 20.0
    translation_threshold_deg: float = 20.0
    rkd_distance_threshold: float = 0.2
    rkd_angle_threshold: float = 0.2
    couple_threshold: float = math.log(1.25)


@dataclass
class TrainConfig:
    max_steps: int = 100
    lr: float = 3e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.15
    clip_grad: float = 1.0
    input_mask_ratio: float = 0.5
    seed: int = 0
    device: str = "cuda"
    amp: bool = True


@dataclass
class ProbeConfig:
    every: int = 20
    decision: bool = False
    min_candidate_step: int = 20
    min_stop_step: int = 40
    patience: int = 3
    min_delta: float = 0.005


LOSS_NAMES = ("feature", "rotation", "translation", "rkd", "couple")


@dataclass
class ProtocolConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def validate(self):
        s, t, p = self.sampling, self.train, self.probe
        if not s.teacher_frames > s.student_frames >= 2:
            raise ValueError("require teacher_frames > student_frames >= 2")
        if (
            min(s.train_tasks, s.probe_tasks, s.dense_candidates, s.dense_min_frames)
            < 1
        ):
            raise ValueError("task counts and sampling limits must be positive")
        if s.sift_mode not in ("auto", "off") or not 0 <= s.sift_threshold <= 1:
            raise ValueError("invalid SIFT configuration")
        if self.model.name not in ("da3", "vggt", "omega", "pi3", "dvlt"):
            raise ValueError("unknown model")
        if self.model.rank < 1 or self.model.alpha <= 0 or self.model.dropout != 0:
            raise ValueError("invalid LoRA configuration")
        if self.model.resolution < 0:
            raise ValueError("resolution must be nonnegative")
        weights = [getattr(self.loss, k) for k in LOSS_NAMES]
        if min(weights) < 0 or not any(weights):
            raise ValueError(
                "loss weights must be nonnegative with at least one active term"
            )
        if self.loss.rkd_delta <= 0 or not 0 < self.loss.rotation_knee_deg < 180:
            raise ValueError("invalid robust loss threshold")
        for k, v in asdict(self.reliability).items():
            if "threshold" in k and v <= 0:
                raise ValueError(f"{k} must be positive")
        if t.max_steps < 1 or t.lr <= 0 or t.weight_decay < 0 or t.clip_grad <= 0:
            raise ValueError("invalid optimizer configuration")
        if not 0 <= t.warmup_ratio < 1 or not 0 <= t.input_mask_ratio <= 1:
            raise ValueError("invalid warmup or mask ratio")
        if (
            min(p.every, p.min_candidate_step, p.min_stop_step, p.patience) < 1
            or p.min_delta < 0
        ):
            raise ValueError("invalid probe configuration")
        for section in asdict(self).values():
            if any(
                isinstance(v, float) and not math.isfinite(v) for v in section.values()
            ):
                raise ValueError("configuration contains non-finite number")
        return self

    def to_dict(self):
        return asdict(self)


def from_dict(data):
    config = ProtocolConfig()
    if not isinstance(data, dict):
        raise TypeError("configuration must be a mapping")
    for section, values in data.items():
        if section not in {f.name for f in fields(config)} or not isinstance(
            values, dict
        ):
            raise ValueError(f"unknown/invalid configuration section: {section}")
        obj = getattr(config, section)
        for key, value in values.items():
            if key not in {f.name for f in fields(obj)}:
                raise ValueError(f"unknown configuration key: {section}.{key}")
            default = getattr(obj, key)
            if not (
                type(value) is type(default)
                or (type(default) is float and type(value) is int)
            ):
                raise ValueError(f"wrong type for {section}.{key}")
            setattr(obj, key, float(value) if type(default) is float else value)
    return config.validate()


def load_config(path=None, overrides=None):
    import yaml

    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = data or {}
    for dotted, value in (overrides or {}).items():
        section, key = dotted.split(".", 1)
        data.setdefault(section, {})[key] = value
    return from_dict(data)
