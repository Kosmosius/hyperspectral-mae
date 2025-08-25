from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Protocol

import torch
from torch import nn, Tensor


# ----------------- Protocols (import-light) -----------------

class EncoderProtocol(Protocol):
    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # returns z [B,d]
        ...


class DecoderProtocol(Protocol):
    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:  # (f_hat[K], c_hat[K]|None)
        ...


# ----------------- SpectralAE -----------------

@dataclass
class SpectralAEConfig:
    """Minimal config for wiring the model."""
    allow_no_continuum: bool = True
    clamp_output: Tuple[float, float] = (-0.05, 1.05)


class SpectralAE(nn.Module):
    """
    End-to-end spectral autoencoder:

        z = Encoder(tokens, mask)
        (f_hat, c_hat) = Decoder(z)
        y_hat = R @ (f_hat * c_hat)      # if c_hat is not None else R @ f_hat

    Shapes:
        tokens:   [B, T, D_tkn]
        mask:     [B, T]  (True=valid)
        f_hat:    [B, K]
        c_hat:    [B, K] | None
        R:        [B, N, K] | [N, K]
        y_hat:    [B, N]
    """
    def __init__(self, encoder: EncoderProtocol, decoder: DecoderProtocol, cfg: SpectralAEConfig = SpectralAEConfig()):
        super().__init__()
        self.encoder = encoder  # type: ignore[assignment]
        self.decoder = decoder  # type: ignore[assignment]
        self.cfg = cfg

    @staticmethod
    def _render_batched(
        f_hat: Tensor,  # [B,K]
        c_hat: Optional[Tensor],  # [B,K] or None
        R: Tensor,      # [B,N,K] or [N,K]
    ) -> Tensor:
        g = f_hat
        if c_hat is not None:
            g = g * c_hat
        if R.dim() == 2:
            # [N,K] x [B,K] -> [B,N]
            y = torch.einsum("nk,bk->bn", R, g)
        elif R.dim() == 3:
            # [B,N,K] x [B,K] -> [B,N]
            y = torch.einsum("bnk,bk->bn", R, g)
        else:
            raise ValueError("R must be [N,K] or [B,N,K]")
        return y

    def forward(
        self,
        tokens: Tensor,
        token_mask: Optional[Tensor],
        R: Tensor,
    ) -> Dict[str, Tensor | None]:
        """
        Forward pass that does *not* compute losses (losses live in ml/losses/*).

        Args:
            tokens: [B,T,D_tkn]
            token_mask: [B,T] (True=valid)
            R: [N,K] or [B,N,K]

        Returns:
            dict with keys: 'z', 'f_hat', 'c_hat', 'y_hat'
        """
        z = self.encoder(tokens, token_mask)  # [B,d]
        f_hat, c_hat = self.decoder(z)        # [B,K], [B,K]|None

        if c_hat is None and not self.cfg.allow_no_continuum:
            raise RuntimeError("Decoder returned no continuum but allow_no_continuum=False")

        f_hat = f_hat.clamp(*self.cfg.clamp_output)
        if c_hat is not None:
            c_hat = c_hat.clamp(0.5, 1.5)  # safety clamp; decoder should already bound this

        y_hat = self._render_batched(f_hat, c_hat, R)  # [B,N]

        return {"z": z, "f_hat": f_hat, "c_hat": c_hat, "y_hat": y_hat}
