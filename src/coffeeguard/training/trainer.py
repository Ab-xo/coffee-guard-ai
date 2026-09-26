"""Multi-stage trainer (LP-FT ready).

Each stage (e.g. ``head`` then ``finetune``) gets its own optimizer (AdamW, optional
layer-wise LR decay), a warmup + cosine schedule, an EMA of the weights, and early
stopping on validation macro-F1. The next stage starts from the previous stage's best
weights. The single best checkpoint across stages is saved as ``best.pt`` with all
metadata needed to evaluate and export it without the training config.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from coffeeguard.config import DataConfig, StageConfig, TrainConfig, dump_config, load_config
from coffeeguard.data.dataset import LeafDataset, read_split
from coffeeguard.data.transforms import eval_transform, train_transform
from coffeeguard.evaluation.metrics import core_metrics
from coffeeguard.models.factory import (
    build_model,
    freeze_frozen_batchnorm,
    model_data_config,
    set_trainable,
)
from coffeeguard.utils.io import read_json, write_json
from coffeeguard.utils.log import get_logger
from coffeeguard.utils.runs import capture_env, create_run_dir
from coffeeguard.utils.seed import set_seed

log = get_logger(__name__)


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def warmup_cosine(step: int, warmup: int, total: int, floor: float) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to ``floor``."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
    max_batches: int | None = None,
    channels_last: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities and labels for a loader."""
    model.eval()
    probs, labels = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _val_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    m = core_metrics(labels, probs.argmax(1))
    eps = 1e-7
    m["loss"] = float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())
    return m


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.data_cfg = load_config(DataConfig, cfg.data_config)
        self.classes = self.data_cfg.classes
        self.device = resolve_device(cfg.device)
        self.amp = cfg.amp and self.device.type == "cuda"
        self.channels_last = cfg.channels_last and self.device.type == "cuda"

        self.model = build_model(
            cfg.model, len(self.classes), cfg.pretrained, cfg.drop_rate, cfg.drop_path_rate
        )
        dcfg = model_data_config(self.model)
        self.mean, self.std = tuple(dcfg["mean"]), tuple(dcfg["std"])
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)

        splits = self.data_cfg.splits_dir
        root = self.data_cfg.processed_dir
        self.train_df = read_split(splits, "train")
        self.val_df = read_split(splits, "val")
        train_ds = LeafDataset(
            self.train_df,
            root,
            train_transform(cfg.img_size, cfg.augment, self.mean, self.std),
            cfg.preload,
        )
        if cfg.augment.bg_swap_p > 0:
            from coffeeguard.data.bgswap import build_background_swap

            imgs = train_ds._cache or [LeafDataset._load(p) for p in train_ds.paths]
            train_ds.pre_transform = build_background_swap(
                imgs, cfg.augment.bg_swap_p, seed=cfg.seed
            )
            log.info(
                "Background swap p=%.2f with %d donor backgrounds",
                cfg.augment.bg_swap_p,
                len(train_ds.pre_transform.donors),
            )
        val_ds = LeafDataset(
            self.val_df, root, eval_transform(cfg.img_size, self.mean, self.std), cfg.preload
        )
        loader_kw = {
            "num_workers": cfg.num_workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": cfg.num_workers > 0,
        }
        g = torch.Generator().manual_seed(cfg.seed)
        self.train_loader = DataLoader(
            train_ds,
            cfg.batch_size,
            shuffle=True,
            generator=g,
            drop_last=len(train_ds) > cfg.batch_size,
            **loader_kw,
        )
        self.val_loader = DataLoader(val_ds, cfg.batch_size * 2, shuffle=False, **loader_kw)

        weight = None
        if cfg.class_weights:
            from sklearn.utils.class_weight import compute_class_weight

            y = self.train_df["class_id"].to_numpy()
            w = compute_class_weight("balanced", classes=np.arange(len(self.classes)), y=y)
            weight = torch.tensor(w, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

        self.run_dir: Path | None = None
        self.best_score = -1.0
        self.best_info: dict = {}

    # ------------------------------------------------------------------ helpers

    def _fingerprint(self) -> str | None:
        info = self.data_cfg.splits_dir / "split_info.json"
        return read_json(info).get("fingerprint") if info.exists() else None

    def _checkpoint(self, state_dict: dict, info: dict) -> dict:
        return {
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            "model": self.cfg.model,
            "classes": self.classes,
            "img_size": self.cfg.img_size,
            "mean": self.mean,
            "std": self.std,
            "data_fingerprint": self._fingerprint(),
            "config": self.cfg.model_dump(mode="json"),
            **info,
        }

    def _log_history(self, row: dict) -> None:
        assert self.run_dir is not None
        with open(self.run_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ training

    def _train_epoch(self, optimizer, scheduler, scaler, ema) -> dict[str, float]:
        cfg = self.cfg
        self.model.train()
        freeze_frozen_batchnorm(self.model)
        total_loss, correct, seen = 0.0, 0, 0
        for i, (x, y) in enumerate(self.train_loader):
            if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            self._step += 1
            if ema is not None:
                ema.update(self.model, step=self._step)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}

    def _run_stage(self, stage: StageConfig) -> None:
        from timm.optim import create_optimizer_v2
        from timm.utils import ModelEmaV3

        cfg = self.cfg
        trainable, total = set_trainable(self.model, stage.trainable, stage.last_n_stages)
        log.info(
            "Stage %s: %s trainable params %.2fM / %.2fM",
            stage.name,
            stage.trainable,
            trainable / 1e6,
            total / 1e6,
        )
        optimizer = create_optimizer_v2(
            self.model,
            opt="adamw",
            lr=stage.lr,
            weight_decay=stage.weight_decay,
            layer_decay=stage.layer_decay,
        )
        steps = len(self.train_loader)
        if cfg.max_train_batches is not None:
            steps = min(steps, cfg.max_train_batches)
        total_steps = max(1, stage.epochs * steps)
        warmup = int(stage.warmup_epochs * steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: warmup_cosine(s, warmup, total_steps, stage.min_lr_ratio)
        )
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        # use_warmup ramps the decay up over the first steps so the EMA isn't stuck
        # near the stage's starting weights when epochs are short.
        ema = (
            ModelEmaV3(self.model, decay=cfg.ema_decay, use_warmup=True) if cfg.ema_decay else None
        )
        self._step = 0

        stage_best, stage_state, bad = -1.0, None, 0
        for epoch in range(1, stage.epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch(optimizer, scheduler, scaler, ema)
            candidates = {"raw": self.model}
            if ema is not None:
                candidates["ema"] = ema.module
            val = {}
            for kind, net in candidates.items():
                probs, labels = predict(
                    net,
                    self.val_loader,
                    self.device,
                    self.amp,
                    cfg.max_eval_batches,
                    self.channels_last,
                )
                val[kind] = _val_metrics(probs, labels)
            kind = max(val, key=lambda k: val[k]["macro_f1"])
            score = val[kind]["macro_f1"]
            row = {
                "stage": stage.name,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_m["loss"],
                "train_acc": train_m["accuracy"],
                **{f"val_{k}": v for k, v in val["raw"].items()},
                **({f"ema_val_{k}": v for k, v in val["ema"].items()} if "ema" in val else {}),
                "selected": kind,
                "seconds": round(time.time() - t0, 1),
            }
            self._log_history(row)
            log.info(
                "[%s %d/%d] train loss %.4f acc %.4f | val macro-F1 raw %.4f%s | %.0fs",
                stage.name,
                epoch,
                stage.epochs,
                train_m["loss"],
                train_m["accuracy"],
                val["raw"]["macro_f1"],
                f" ema {val['ema']['macro_f1']:.4f}" if "ema" in val else "",
                row["seconds"],
            )

            if score > stage_best:
                stage_best, bad = score, 0
                stage_state = copy.deepcopy(candidates[kind].state_dict())
                if score > self.best_score:
                    self.best_score = score
                    self.best_info = {
                        "stage": stage.name,
                        "epoch": epoch,
                        "weights": kind,
                        "val": val[kind],
                    }
                    torch.save(
                        self._checkpoint(stage_state, self.best_info), self.run_dir / "best.pt"
                    )
            else:
                bad += 1
                if bad >= stage.patience:
                    log.info("Early stopping stage %s at epoch %d", stage.name, epoch)
                    break
        if stage_state is not None:  # next stage starts from this stage's best weights
            self.model.load_state_dict(stage_state)

    def fit(self) -> Path:
        cfg = self.cfg
        self.run_dir = create_run_dir(f"{cfg.name}-s{cfg.seed}", cfg.runs_dir)
        dump_config(cfg, self.run_dir / "config.yaml")
        env = capture_env()
        env["data_fingerprint"] = self._fingerprint()
        env["device"] = str(self.device)
        write_json(self.run_dir / "env.json", env)
        log.info(
            "Run dir: %s | device %s | train %d | val %d",
            self.run_dir,
            self.device,
            len(self.train_df),
            len(self.val_df),
        )
        t0 = time.time()
        for stage in cfg.stages:
            self._run_stage(stage)
        summary = {
            "best_val_macro_f1": self.best_score,
            **self.best_info,
            "minutes": round((time.time() - t0) / 60, 2),
        }
        write_json(self.run_dir / "summary.json", summary)
        log.info(
            "Done. Best val macro-F1 %.4f (%s, epoch %s, %s weights)",
            self.best_score,
            self.best_info.get("stage"),
            self.best_info.get("epoch"),
            self.best_info.get("weights"),
        )
        return self.run_dir


def train(cfg: TrainConfig) -> Path:
    return Trainer(cfg).fit()
