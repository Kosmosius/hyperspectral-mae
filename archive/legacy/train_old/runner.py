from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..ml.model import SpectralAE
from ..ml.render_torch import render_bandspace
from ..ml.losses.band_likelihood import (
    hetero_huber_nll,
    HeteroHuberConfig,
    expected_calibration_error,
)
from ..ml.losses.spectral_losses import (
    sam_loss,
    curvature_loss,
    CurvatureConfig,
    box_penalty,
    violation_rate,
    cr_loss_smooth,
)
from ..ml.losses.invariance import SensorAdversary, hsic_rbf, mmd_rbf
from .schedules import grl_lambda_schedule

# Optional: Weights & Biases (graceful fallback)
try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


# ------------------------------ Configs ------------------------------

@dataclass
class SigmaParams:
    sigma0: float = 0.002
    sigma_read: float = 0.004
    sigma_scene: float = 0.02


@dataclass
class LossWeights:
    band: float = 1.0
    canon_mse: float = 0.5    # if f_true provided
    sam: float = 0.5
    cr: float = 0.1
    curvature: float = 1e-4
    box: float = 1e-4
    multiview: float = 0.1
    adv: float = 0.0
    hsic: float = 0.0
    mmd: float = 0.0


@dataclass
class RunnerConfig:
    epochs: int = 150
    log_interval: int = 50
    val_interval: int = 1
    grad_accum: int = 1
    amp: bool = True
    grad_clip: float = 1.0
    ckpt_dir: str = "./checkpoints"
    save_best_on: str = "val/band_mae"
    find_unused_params: bool = False   # set True if you enable/disable heads dynamically
    adversary_classes: int = 0        # 0 disables adversary
    grl_warmup_start: int = 30
    grl_warmup_end: int = 100
    grl_max_lambda: float = 0.05


# ------------------------------ Helpers ------------------------------

def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0")) if torch.distributed.is_available() and torch.distributed.is_initialized() else 0


def ddp_world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1")) if torch.distributed.is_available() and torch.distributed.is_initialized() else 1


def is_master() -> bool:
    return ddp_rank() == 0


def setup_wandb(run_name: str, cfg: Dict[str, Any]) -> None:
    if not _WANDB or not is_master():
        return
    wandb.init(project=cfg.get("project", "spectralae"), name=run_name, config=cfg, mode=cfg.get("mode", "online"))


def log_metrics(metrics: Dict[str, float], step: int) -> None:
    if _WANDB and is_master():
        wandb.log(metrics, step=step)
    elif is_master():
        # minimal stdout logger
        print(json.dumps({"step": step, **metrics}))


def save_ckpt(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_ckpt(path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", 0))


def set_grl_lambda(model: nn.Module, value: float) -> None:
    for m in model.modules():
        if hasattr(m, "grl") and hasattr(m.grl, "lamb"):
            m.grl.lamb = float(value)


# ------------------------------ Runner ------------------------------

class Runner:
    """
    Orchestrates training and validation.

    Expected batch structure (flexible; provide what you have):
      - tokens: [B,T,D]
      - token_mask: [B,T] boolean (True=valid)
      - R: [B,N,K] or [N,K]
      - y: [B,N]                      (band measurements)
      - f_true: [B,K] (optional)      (canonical target)
      - srf_summary: [B,S] (optional) (SRF features for HSIC/MMD)
      - domain_id: [B] int (optional) (for adversary classification)
      - (optional multi-view): tokens_b, token_mask_b, R_b, y_b

    The SpectralAE model returns: z, f_hat, c_hat, y_hat
    """
    def __init__(
        self,
        model: SpectralAE,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        run_cfg: RunnerConfig = RunnerConfig(),
        loss_w: LossWeights = LossWeights(),
        sigma_params: SigmaParams = SigmaParams(),
        adversary: Optional[SensorAdversary] = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_cfg = run_cfg
        self.loss_w = loss_w
        self.sigma_params = sigma_params
        self.adversary = adversary.to(device) if adversary is not None else None

        if ddp_world() > 1:
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=run_cfg.find_unused_params,
                broadcast_buffers=False,
            )
            if self.adversary is not None:
                self.adversary = DDP(
                    self.adversary,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    output_device=device.index if device.type == "cuda" else None,
                    broadcast_buffers=False,
                )

        self.scaler = torch.cuda.amp.GradScaler(enabled=run_cfg.amp)

        # GRL schedule
        self._grl_sched = grl_lambda_schedule(
            run_cfg.grl_warmup_start, run_cfg.grl_warmup_end, run_cfg.grl_max_lambda, ramp="linear"
        )

        self.global_step = 0
        self.best_metric = float("inf")

    # ---------- Sigma model ----------

    def _compute_sigma(self, R: Tensor, y_hat: Tensor) -> Tensor:
        """
        sigma^2 = sigma0^2 + sigma_read^2 / EW + sigma_scene^2 * y_hat^2
        EW (effective width) = 1 / sum_k r_{ik}^2
        Broadcasts across batch.
        """
        if R.dim() == 2:
            # [N,K]
            ew = 1.0 / (R.pow(2).sum(dim=-1).clamp_min(1e-12))  # [N]
            ew = ew.unsqueeze(0).expand(y_hat.shape[0], -1)     # [B,N]
        else:
            # [B,N,K]
            ew = 1.0 / (R.pow(2).sum(dim=-1).clamp_min(1e-12))  # [B,N]

        sp = self.sigma_params
        sigma2 = (sp.sigma0 ** 2) + (sp.sigma_read ** 2) / ew + (sp.sigma_scene ** 2) * (y_hat ** 2)
        return sigma2.sqrt()

    # ---------- Losses ----------

    def _compute_losses(self, batch: Dict[str, Tensor], out: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        y = batch["y"].to(self.device)                            # [B,N]
        R = batch["R"].to(self.device)                            # [N,K] or [B,N,K]
        f_true = batch.get("f_true")
        if f_true is not None:
            f_true = f_true.to(self.device)                       # [B,K]

        # Band NLL (heteroscedastic Huber)
        sigma = self._compute_sigma(R, out["y_hat"])              # [B,N]
        band_resid = out["y_hat"] - y                             # [B,N]
        band_loss = hetero_huber_nll(band_resid, sigma, HeteroHuberConfig(kappa=1.0, include_log_sigma=True))

        # Canonical losses
        canon_mse = torch.tensor(0.0, device=self.device)
        sam = torch.tensor(0.0, device=self.device)
        cr = torch.tensor(0.0, device=self.device)
        curv = torch.tensor(0.0, device=self.device)
        box = torch.tensor(0.0, device=self.device)
        nvr = torch.tensor(0.0, device=self.device)

        if f_true is not None:
            canon_mse = torch.mean((out["f_hat"] - f_true) ** 2)
            sam = sam_loss(out["f_hat"], f_true, use_angle=False)
            cr = cr_loss_smooth(out["f_hat"], f_true)
            curv = curvature_loss(out["f_hat"], CurvatureConfig(lambda_weights=None))
            box = box_penalty(out["f_hat"])
            nvr = violation_rate(out["f_hat"])

        # Multi-view consistency (if view_b exists)
        mv = torch.tensor(0.0, device=self.device)
        if "z_b" in out and out["z_b"] is not None:
            mv = (out["z"] - out["z_b"]).pow(2).mean()
            if "f_hat_b" in out and out["f_hat_b"] is not None:
                mv = mv + (out["f_hat"] - out["f_hat_b"]).pow(2).mean()

        # Adversarial / independence
        adv = torch.tensor(0.0, device=self.device)
        adv_acc = torch.tensor(0.0, device=self.device)
        if self.adversary is not None and "domain_id" in batch and self.run_cfg.adversary_classes > 0:
            set_grl_lambda(self.adversary, self._grl_sched(self.current_epoch))
            logits = self.adversary(out["z"].detach())  # detach to avoid double-backprop into encoder
            target = batch["domain_id"].to(self.device).long()
            adv = torch.nn.functional.cross_entropy(logits, target)
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                adv_acc = (pred == target).float().mean()

        hsic = torch.tensor(0.0, device=self.device)
        mmd = torch.tensor(0.0, device=self.device)
        if "srf_summary" in batch and batch["srf_summary"] is not None and self.loss_w.hsic + self.loss_w.mmd > 0:
            srf_summ = batch["srf_summary"].to(self.device).float()
            z = out["z"].float()
            if self.loss_w.hsic > 0:
                hsic = hsic_rbf(z, srf_summ)
            if self.loss_w.mmd > 0:
                # Split srf_summ by a simple threshold (e.g., domain partition) is task-specific.
                # Here we treat domain_id as groups if provided; otherwise skip MMD.
                if "domain_id" in batch:
                    d = batch["domain_id"].to(self.device)
                    if d.unique().numel() >= 2:
                        g0 = z[d == d.unique()[0]]
                        g1 = z[d == d.unique()[1]]
                        if g0.numel() > 0 and g1.numel() > 0:
                            mmd = mmd_rbf(g0, g1)

        # Weighted sum
        total = (
            self.loss_w.band * band_loss
            + self.loss_w.canon_mse * canon_mse
            + self.loss_w.sam * sam
            + self.loss_w.cr * cr
            + self.loss_w.curvature * curv
            + self.loss_w.box * box
            + self.loss_w.multiview * mv
            + self.loss_w.adv * adv
            + self.loss_w.hsic * hsic
            + self.loss_w.mmd * mmd
        )

        # Diagnostics
        ece = expected_calibration_error(band_resid.detach(), sigma.detach())
        mae = band_resid.abs().mean()

        metrics = {
            "loss/total": float(total.detach().cpu()),
            "loss/band": float(band_loss.detach().cpu()),
            "loss/canon_mse": float(canon_mse.detach().cpu()),
            "loss/sam": float(sam.detach().cpu()),
            "loss/cr": float(cr.detach().cpu()),
            "loss/curv": float(curv.detach().cpu()),
            "loss/box": float(box.detach().cpu()),
            "loss/mv": float(mv.detach().cpu()),
            "loss/adv": float(adv.detach().cpu()),
            "stat/nvr": float(nvr.detach().cpu()),
            "stat/ece": float(ece.detach().cpu()),
            "band/mae": float(mae.detach().cpu()),
            "adv/acc": float(adv_acc.detach().cpu()),
        }
        return total, metrics

    # ---------- Public train/eval ----------

    def fit(self, start_epoch: int = 0, run_name: str = "run") -> None:
        if is_master():
            os.makedirs(self.run_cfg.ckpt_dir, exist_ok=True)
        setup_wandb(run_name, {"runner": asdict(self.run_cfg), "loss_weights": asdict(self.loss_w)})

        for epoch in range(start_epoch, self.run_cfg.epochs):
            self.current_epoch = epoch
            if ddp_world() > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self._train_one_epoch()
            if is_master():
                log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=self.global_step)

            if self.val_loader is not None and (epoch % self.run_cfg.val_interval == 0):
                val_metrics = self.evaluate(self.val_loader)
                if is_master():
                    log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=self.global_step)
                    # Track best
                    key = self.run_cfg.save_best_on.replace("val/", "")
                    current = val_metrics.get(key, float("inf"))
                    if current < self.best_metric:
                        self.best_metric = current
                        save_ckpt(os.path.join(self.run_cfg.ckpt_dir, "best.pt"), self._unwrap(self.model), self.optimizer, epoch, self.scaler)

            if is_master():
                save_ckpt(os.path.join(self.run_cfg.ckpt_dir, "last.pt"), self._unwrap(self.model), self.optimizer, epoch, self.scaler)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self._unwrap(self.model).eval()
        if self.adversary is not None:
            self._unwrap(self.adversary).eval()

        agg = {}
        count = 0
        for batch in loader:
            out = self._forward_batch(batch, train_mode=False)
            _, m = self._compute_losses(batch, out)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            count += 1

        for k in agg:
            agg[k] /= max(1, count)
        return agg

    # ---------- Internals ----------

    def _unwrap(self, maybe_ddp: nn.Module) -> nn.Module:
        return maybe_ddp.module if isinstance(maybe_ddp, DDP) else maybe_ddp

    def _forward_batch(self, batch: Dict[str, Tensor], train_mode: bool = True) -> Dict[str, Tensor]:
        tokens = batch["tokens"].to(self.device)
        token_mask = batch.get("token_mask")
        token_mask = token_mask.to(self.device) if token_mask is not None else None
        R = batch["R"].to(self.device)

        self._unwrap(self.model).train(train_mode)
        with torch.cuda.amp.autocast(enabled=self.run_cfg.amp):
            out = self.model(tokens, token_mask, R)

            # Optional: multi-view inputs (suffix _b)
            if all(k in batch for k in ("tokens_b", "R_b")):
                tokens_b = batch["tokens_b"].to(self.device)
                token_mask_b = batch.get("token_mask_b")
                token_mask_b = token_mask_b.to(self.device) if token_mask_b is not None else None
                R_b = batch["R_b"].to(self.device)
                out_b = self.model(tokens_b, token_mask_b, R_b)
                out["z_b"] = out_b["z"]
                out["f_hat_b"] = out_b["f_hat"]

        return out

    def _train_one_epoch(self) -> Dict[str, float]:
        self._unwrap(self.model).train(True)
        if self.adversary is not None:
            self._unwrap(self.adversary).train(True)

        meter = {}
        self.optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(self.train_loader):
            out = self._forward_batch(batch, train_mode=True)
            total_loss, metrics = self._compute_losses(batch, out)

            # Log running metrics
            for k, v in metrics.items():
                meter[k] = meter.get(k, 0.0) + float(v)

            # Backprop with AMP + grad accumulation
            self.scaler.scale(total_loss / self.run_cfg.grad_accum).backward()

            if (it + 1) % self.run_cfg.grad_accum == 0:
                if self.run_cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._unwrap(self.model).parameters(), self.run_cfg.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if is_master() and (it + 1) % self.run_cfg.log_interval == 0:
                avg = {k: v / self.run_cfg.log_interval for k, v in meter.items()}
                log_metrics({f"iter/{k}": x for k, x in avg.items()}, step=self.global_step)
                meter = {}

            self.global_step += 1

        # Epoch averages (approximate)
        steps = max(1, len(self.train_loader))
        for k in meter:
            meter[k] /= steps
        return meter
