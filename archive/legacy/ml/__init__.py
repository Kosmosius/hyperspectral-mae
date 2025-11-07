"""
PyTorch ML stack for SpectralAE (Aim 1).

Contents
--------
- interfaces: Protocols/ABCs for Encoders/Decoders/Models (framework-level contracts)
- modules.encoders: AdapterTransformer, SetTransformer, RelPosRFFBias

Design choices
--------------
- Pre-LN Transformer blocks with residual LayerScale (optional) for stability.
- Attention module supports an additive attention bias tensor for relative spectral bias.
- Encoders accept masks (B,T) and optional center wavelengths (nm or normalized).
"""

from .interfaces import (
    EncoderProtocol,
    DecoderProtocol,
    SpectralModelProtocol,
    EncoderOutput,
)

from .modules.encoders.adapter_transformer import (
    AdapterTransformerConfig,
    AdapterTransformerEncoder,
)

from .modules.encoders.set_transformer import (
    SetTransformerConfig,
    SetTransformerEncoder,
)

from .modules.encoders.relpos_bias import (
    RelPosRFFBiasConfig,
    RelPosRFFBias,
)

__all__ = [
    # interfaces
    "EncoderProtocol",
    "DecoderProtocol",
    "SpectralModelProtocol",
    "EncoderOutput",
    # encoders
    "AdapterTransformerConfig",
    "AdapterTransformerEncoder",
    "SetTransformerConfig",
    "SetTransformerEncoder",
    # relpos
    "RelPosRFFBiasConfig",
    "RelPosRFFBias",
]
