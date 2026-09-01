from __future__ import annotations

from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf


def initialize_wandb(cfg: DictConfig, run_dir: Path):
    """Initialize W&B with the fully resolved Hydra configuration."""
    return wandb.init(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.get("entity"),
        name=cfg.wandb.get("name") or cfg.run.get("name"),
        group=cfg.wandb.get("group"),
        tags=list(cfg.wandb.get("tags", [])),
        mode=str(cfg.wandb.mode),
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

