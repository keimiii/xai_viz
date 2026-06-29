"""DINOv2 model wrapper with register tokens.

DINOv2 (DINO version 2) is a self-supervised ViT trained with a combination
of self-distillation and other techniques. The "with-registers" variant adds
4 learnable register tokens that absorb high-norm artifacts.

Sequence structure: [CLS] + [4 registers] + [256 patches]
- Patch size: 14 (so 224/14 = 16 patches per side = 256 total)
- Register tokens: 4 (positioned after CLS, before patches)
"""

from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from ssl_attention.config import MODELS
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

# Load configuration from central config
_config = MODELS["dinov2"]


class DINOv2(BaseVisionModel):
    """DINOv2 with registers wrapper.

    Uses facebook/dinov2-with-registers-base which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 14x14 patches (256 patches for 224x224 images)
    - 4 register tokens

    Example:
        >>> model = DINOv2()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 256, 768)
    """

    model_name = "dinov2"
    model_id = _config.model_id
    patch_size = _config.patch_size
    embed_dim = _config.embed_dim
    num_layers = _config.num_layers
    num_heads = _config.num_heads
    num_registers = _config.num_registers

    def _load_model(self) -> nn.Module:
        """Load DINOv2 with registers from HuggingFace.

        Uses AutoModel to correctly load the dinov2_with_registers variant.
        """
        config = self._load_config()
        return AutoModel.from_pretrained(self.model_id, config=config)  # type: ignore[no-any-return]

    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Extract standardized output from DINOv2 output.

        DINOv2 returns:
        - last_hidden_state: (B, seq_len, D) where seq_len = 1 + 4 + 256 = 261
        - attentions: tuple of L tensors, each (B, H, seq, seq)
        - hidden_states: tuple of L+1 tensors when output_hidden_states=True

        We extract:
        - CLS token: position 0
        - Patch tokens: positions 5 onwards (skip CLS + 4 registers)
        """
        last_hidden = model_output.last_hidden_state  # (B, 261, 768)
        attentions = model_output.attentions  # tuple of 12 tensors

        # CLS is always position 0
        cls_token = last_hidden[:, 0, :]  # (B, 768)

        # Patches start after CLS + registers
        patch_start = 1 + self.num_registers  # 5
        patch_tokens = last_hidden[:, patch_start:, :]  # (B, 256, 768)

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


# Convenience function for quick instantiation
def create_dinov2(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DINOv2:
    """Create a DINOv2 model instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured DINOv2 model.
    """
    return DINOv2(device=device, dtype=dtype)
