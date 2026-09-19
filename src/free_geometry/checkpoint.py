"""Checkpoint persistence and an optional, replayable probe policy."""

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


def config_fingerprint(config):
    return hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode()
    ).hexdigest()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass
class CheckpointPolicy:
    best_step: int = 0
    best_score: float = math.inf
    significant_score: float = math.inf
    stale: int = 0

    def observe(self, step, score, config):
        if not config.decision or step < config.min_candidate_step:
            return False
        valid = score is not None and math.isfinite(score)
        if valid:
            if score < self.best_score:
                self.best_score, self.best_step = score, step
            improvement = self.significant_score - score
            if not math.isfinite(
                self.significant_score
            ) or improvement > config.min_delta * max(
                abs(self.significant_score), 1e-12
            ):
                self.significant_score, self.stale = score, 0
            else:
                self.stale += 1
        else:
            self.stale += 1
        return step >= config.min_stop_step and self.stale >= config.patience

    def state_dict(self):
        return asdict(self)


def load_checkpoint(path, config, manifest, identity=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("version") != "unified-1"
        or payload["manifest"] != manifest["fingerprint"]
    ):
        raise ValueError("checkpoint manifest/version mismatch")
    if payload["config_hash"] != config_fingerprint(config):
        raise ValueError("checkpoint configuration mismatch")
    if identity is not None and payload["model_identity"] != identity:
        raise ValueError("checkpoint model weights/source mismatch")
    return payload


def model_identity(config, resolved_weights=None):
    """Content-address local weights; also record external source Git revision."""
    import subprocess

    path = Path(resolved_weights or config.model.weights)
    result = {"model": config.model.name, "weights": config.model.weights, "files": {}}
    files = (
        [path]
        if path.is_file()
        else sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix in (".pt", ".pth", ".bin", ".safetensors", ".json")
        )
        if path.is_dir() and (resolved_weights or config.model.weights)
        else []
    )
    for file in files:
        h = hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        result["files"][str(file.relative_to(path) if path.is_dir() else file.name)] = (
            h.hexdigest()
        )
    if config.model.source:
        proc = subprocess.run(
            ["git", "-C", config.model.source, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        result["source_revision"] = (
            proc.stdout.strip() if proc.returncode == 0 else None
        )
    return result
