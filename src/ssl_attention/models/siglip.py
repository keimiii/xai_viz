"""SigLIP (v1) vision encoder wrapper.

SigLIP (Sigmoid Loss for Language-Image Pre-training) replaces CLIP's
softmax-based contrastive loss with a sigmoid loss, enabling better
batch efficiency and multilingual support.

Sequence structure: [196 patches] (NO CLS token in sequence)
- Patch size: 16 (so 224/16 = 14 patches per side = 196 total)
- No register tokens
- Pooler output is derived separately, not from a CLS token
"""

from typing import Any

import torch
from torch import nn
from transformers import AutoImageProcessor, SiglipVisionConfig, SiglipVisionModel

from ssl_attention.config import MODELS
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

# Load configuration from central config
_config = MODELS["siglip"]


class SigLIP(BaseVisionModel):
    """SigLIP (v1) vision encoder wrapper.

    Uses google/siglip-base-patch16-224 which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 16x16 patches (196 patches for 224x224 images)
    - No register tokens
    - NO CLS token in sequence (uses separate pooler)

    Note: SigLIP uses sigmoid loss instead of softmax, which affects
    the learned attention patterns (tends to be more distributed).

    Unlike CLIP/DINOv2/MAE, SigLIP doesn't have a CLS token in the sequence.
    The cls_token output is the pooler_output (MAP attention pooling).

    Example:
        >>> model = SigLIP()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 196, 768)
    """

    model_name = "siglip"
    model_id = _config.model_id
    patch_size = _config.patch_size
    embed_dim = _config.embed_dim
    num_layers = _config.num_layers
    num_heads = _config.num_heads
    num_registers = _config.num_registers

    def _load_processor(self) -> Any:
        """Load SigLIP processor with fixed 224x224 resolution."""
        processor = AutoImageProcessor.from_pretrained(self.model_id)
        # Force fixed 224x224 output to get exactly 196 patches (14x14)
        processor.size = {"height": 224, "width": 224}
        processor.do_resize = True
        return processor

    def _load_config(self) -> SiglipVisionConfig:
        """Load SigLIP vision config with attention output enabled.

        SigLIP uses SiglipVisionConfig, not AutoConfig, for the vision encoder.
        """
        config = SiglipVisionConfig.from_pretrained(self.model_id)
        config.output_attentions = True
        return config

    def _load_model(self) -> nn.Module:
        """Load SigLIP vision encoder from HuggingFace with attention output enabled."""
        config = self._load_config()
        return SiglipVisionModel.from_pretrained(self.model_id, config=config)

    def forward(
        self, images: torch.Tensor, output_hidden_states: bool = False
    ) -> ModelOutput:
        """Forward pass through the SigLIP vision backbone."""
        with self.inference_context():
            model_output = self.model(
                pixel_values=images,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
            )
        return self._extract_output(model_output, output_hidden_states)

    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Extract standardized output from SigLIP vision output.

        SigLIP vision encoder returns:
        - last_hidden_state: (B, 196, D) - NO CLS token in sequence
        - pooler_output: (B, D) - pooled representation (MAP attention pooling)
        - attentions: tuple of L tensors, each (B, H, 196, 196)
        - hidden_states: tuple of L+1 tensors when output_hidden_states=True

        We extract:
        - CLS token: pooler_output (since there's no CLS in sequence)
        - Patch tokens: all of last_hidden_state (all 196 are patches)
        """
        last_hidden = model_output.last_hidden_state  # (B, 196, 768)
        pooler_output = model_output.pooler_output  # (B, 768)
        attentions = model_output.attentions  # tuple of 12 tensors

        # SigLIP has no CLS token - use pooler output
        cls_token = pooler_output  # (B, 768)

        # All positions are patches (no CLS to skip)
        patch_tokens = last_hidden  # (B, 196, 768)

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


def create_siglip(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SigLIP:
    """Create a SigLIP (v1) vision encoder instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured SigLIP model.
    """
    return SigLIP(device=device, dtype=dtype)
