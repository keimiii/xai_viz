"""MAE (Masked Autoencoder) model wrapper.

MAE is trained to reconstruct masked patches from visible patches.
For attention analysis, we keep all patches visible and pass deterministic
forward noise so HuggingFace MAE preserves canonical patch ordering.

Sequence structure: [CLS] + [196 patches]
- Patch size: 16 (so 224/16 = 14 patches per side = 196 total)
- No register tokens
"""

from typing import Any

import torch
from torch import nn
from transformers import ViTMAEConfig, ViTMAEModel

from ssl_attention.config import MODELS
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

# Load configuration from central config
_config = MODELS["mae"]


class MAE(BaseVisionModel):
    """MAE (Masked Autoencoder) wrapper.

    Uses facebook/vit-mae-base which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 16x16 patches (196 patches for 224x224 images)
    - No register tokens

    Note: We set `mask_ratio=0.0` to keep every patch visible. The installed
    HuggingFace MAE implementation still permutes patches unless analysis
    forwards also pass deterministic `noise`, which the shared base wrapper does.

    Example:
        >>> model = MAE()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 196, 768)
    """

    model_name = "mae"
    model_id = _config.model_id
    patch_size = _config.patch_size
    embed_dim = _config.embed_dim
    num_layers = _config.num_layers
    num_heads = _config.num_heads
    num_registers = _config.num_registers

    def _load_config(self) -> ViTMAEConfig:
        """Load MAE config with attention output enabled and all patches visible.

        Analysis forwards still need deterministic `noise` because the
        HuggingFace MAE implementation calls `random_masking()` even when
        `mask_ratio=0.0`.
        """
        config = ViTMAEConfig.from_pretrained(self.model_id)
        config.output_attentions = True
        config.mask_ratio = 0.0  # Keep all patches visible for analysis forwards
        return config

    def _load_model(self) -> nn.Module:
        """Load MAE from HuggingFace with attention output enabled."""
        config = self._load_config()
        return ViTMAEModel.from_pretrained(self.model_id, config=config)

    # Note: forward() is inherited from base class, which routes MAE analysis
    # through the deterministic-noise helper to keep patch order stable.

    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Extract standardized output from MAE output.

        MAE returns:
        - last_hidden_state: (B, seq_len, D) where seq_len = 1 + 196 = 197
        - attentions: tuple of L tensors, each (B, H, seq, seq)
        - hidden_states: tuple of L+1 tensors when output_hidden_states=True

        We extract:
        - CLS token: position 0
        - Patch tokens: positions 1 onwards (no registers)
        """
        last_hidden = model_output.last_hidden_state  # (B, 197, 768)
        attentions = model_output.attentions  # tuple of 12 tensors

        # CLS is position 0
        cls_token = last_hidden[:, 0, :]  # (B, 768)

        # Patches start at position 1 (no registers)
        patch_tokens = last_hidden[:, 1:, :]  # (B, 196, 768)

        # Convert attention tuple to list
        attention_weights = list(attentions)

        # Extract per-layer hidden states if requested
        # hidden_states[0] is post-embedding, [1:] are post-transformer-layer
        all_hidden_states = None
        if include_hidden_states and model_output.hidden_states is not None:
            # Skip embedding layer (index 0), keep transformer layer outputs
            all_hidden_states = list(model_output.hidden_states[1:])

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attention_weights,
            hidden_states=all_hidden_states,
        )


def create_mae(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> MAE:
    """Create a MAE model instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured MAE model.
    """
    return MAE(device=device, dtype=dtype)
