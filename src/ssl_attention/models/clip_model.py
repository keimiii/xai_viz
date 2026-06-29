"""CLIP vision encoder wrapper.

CLIP (Contrastive Language-Image Pretraining) is trained to align image
and text representations. We use only the vision encoder for attention
analysis, extracting attention patterns from its ViT backbone.

Sequence structure: [CLS] + [196 patches]
- Patch size: 16 (so 224/16 = 14 patches per side = 196 total)
- No register tokens

Note: File is named clip_model.py to avoid conflict with the 'clip' package.
"""

from typing import Any

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from ssl_attention.config import MODELS
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

# Load configuration from central config
_config = MODELS["clip"]


class CLIP(BaseVisionModel):
    """CLIP vision encoder wrapper.

    Uses openai/clip-vit-base-patch16 which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 16x16 patches (196 patches for 224x224 images)
    - No register tokens

    This wraps only the vision encoder, not the full CLIP model.

    Example:
        >>> model = CLIP()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 196, 768)
    """

    model_name = "clip"
    model_id = _config.model_id
    patch_size = _config.patch_size
    embed_dim = _config.embed_dim
    num_layers = _config.num_layers
    num_heads = _config.num_heads
    num_registers = _config.num_registers

    def _load_config(self) -> CLIPVisionConfig:
        """Load CLIP vision config with attention output enabled.

        CLIP uses CLIPVisionConfig, not AutoConfig, for the vision encoder.
        """
        config = CLIPVisionConfig.from_pretrained(self.model_id)
        config.output_attentions = True
        return config

    def _load_model(self) -> nn.Module:
        """Load CLIP vision encoder from HuggingFace with attention output enabled."""
        config = self._load_config()
        return CLIPVisionModel.from_pretrained(self.model_id, config=config)

    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Extract standardized output from CLIP vision output.

        CLIP vision encoder returns:
        - last_hidden_state: (B, seq_len, D) where seq_len = 1 + 196 = 197
        - pooler_output: (B, D) - CLS token after projection
        - attentions: tuple of L tensors, each (B, H, seq, seq)
        - hidden_states: tuple of L+1 tensors when output_hidden_states=True

        We extract:
        - CLS token: position 0 of last_hidden_state (pre-projection)
        - Patch tokens: positions 1 onwards
        """
        last_hidden = model_output.last_hidden_state  # (B, 197, 768)
        attentions = model_output.attentions  # tuple of 12 tensors

        # CLS is position 0
        cls_token = last_hidden[:, 0, :]  # (B, 768)

        # Patches start at position 1
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


def create_clip(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> CLIP:
    """Create a CLIP vision encoder instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured CLIP model.
    """
    return CLIP(device=device, dtype=dtype)
