from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable, Tuple

import torch
from torch import Tensor


@dataclass
class EncoderOutput:
    """
    Output container for encoders.

    Attributes:
        z:           [B, d] pooled latent embedding (for retrieval / downstream)
        tokens_out:  [B, T, d_model] token representations after the stack
        attn_maps:   Optional[List[Tensor]] attention maps for debugging (optional)
        mask:        [B, T] boolean mask echoed back (True=keep, False=pad)
    """
    z: Tensor
    tokens_out: Tensor
    attn_maps: Optional[list[Tensor]]
    mask: Optional[Tensor]


@runtime_checkable
class EncoderProtocol(Protocol):
    """
    Common contract for spectral encoders.
    """

    def forward(
        self,
        tokens: Tensor,                    # [B, T, D_in]
        mask: Optional[Tensor] = None,     # [B, T] True=keep, False=pad
        centers_nm: Optional[Tensor] = None,   # [B, T] (float), optional for rel-pos bias
        centers01: Optional[Tensor] = None,     # [B, T] normalized to [0,1], alt to centers_nm
    ) -> EncoderOutput: ...
    

@runtime_checkable
class DecoderProtocol(Protocol):
    """
    Minimal contract for a spectral decoder in Aim 1:

    Given a latent vector z, produce decoded canonical reflectance f_hat[K]
    and optionally a continuum c_hat[K].
    """

    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            f_hat: [B, K]
            c_hat: Optional[Tensor] [B, K] or None
        """
        ...


@runtime_checkable
class SpectralModelProtocol(Protocol):
    """
    (Optional) End-to-end model protocol to combine encoder+decoder with rendering hooks
    for training loops. Not required for this drop, but defined here for completeness.
    """

    def encode(self, tokens: Tensor, mask: Optional[Tensor] = None,
               centers_nm: Optional[Tensor] = None, centers01: Optional[Tensor] = None) -> EncoderOutput: ...

    def decode(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]: ...
