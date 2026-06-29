"""SigLIP 2 vision encoder wrapper.

SigLIP (Sigmoid Loss for Language-Image Pre-training) replaces CLIP's
softmax-based contrastive loss with a sigmoid loss. SigLIP 2 is the
second generation with improved training and architecture.

Sequence structure: [196 patches] (NO CLS token in sequence)
- Patch size: 16 (so 224/16 = 14 patches per side = 196 total)
- No register tokens
- Pooler output is derived separately, not from a CLS token

Note: SigLIP 2's default image processor uses dynamic aspect-ratio-preserving
resizing (NaFlex-style), which can produce non-square patch grids. We override
this to force fixed 224x224 preprocessing for consistent 14x14 patch grids.
"""

from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    Siglip2VisionConfig,
    Siglip2VisionModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from ssl_attention.config import MODELS
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

_config = MODELS["siglip2"]


class SigLIP2(BaseVisionModel):
    """SigLIP 2 vision encoder wrapper.

    Uses google/siglip2-base-patch16-224 which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 16x16 patches (196 patches for 224x224 images)
    - No register tokens
    - NO CLS token in sequence (uses separate pooler)

    Note: SigLIP uses sigmoid loss instead of softmax, which affects
    the learned attention patterns (tends to be more distributed).

    Unlike CLIP/DINOv2/MAE, SigLIP2 doesn't have a CLS token in the sequence.
    The cls_token output is the pooler_output (MAP attention pooling).

    Example:
        >>> model = SigLIP2()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 196, 768)
    """

    model_name = "siglip2"
    model_id = _config.model_id
    patch_size = _config.patch_size
    embed_dim = _config.embed_dim
    num_layers = _config.num_layers
    num_heads = _config.num_heads
    num_registers = _config.num_registers

    def _load_processor(self) -> Any:
        """Load SigLIP 2 processor with fixed 224x224 resolution.

        SigLIP 2's default processor uses dynamic aspect-ratio-preserving
        resizing (NaFlex-style), which can produce non-square patch grids
        (e.g., 13x15 = 195 patches instead of 14x14 = 196).

        We override the default to force fixed 224x224 output for consistent
        14x14 patch grids compatible with attention visualization.
        """
        processor = AutoImageProcessor.from_pretrained(self.model_id)
        processor.size = {"height": 224, "width": 224}
        processor.do_resize = True
        return processor

    def _load_config(self) -> Siglip2VisionConfig | SiglipVisionConfig:
        """Load a compatible SigLIP vision config with attention output enabled.

        Some checkpoints under the SigLIP 2 naming convention still publish
        `model_type='siglip'` configs. We select the matching vision config
        class based on the actual checkpoint metadata to avoid load-time
        shape mismatches.
        """
        auto_config = AutoConfig.from_pretrained(self.model_id)
        model_type = getattr(auto_config, "model_type", None)
        config: Siglip2VisionConfig | SiglipVisionConfig

        if model_type == "siglip2":
            config = Siglip2VisionConfig.from_pretrained(self.model_id)
        elif model_type == "siglip":
            config = SiglipVisionConfig.from_pretrained(self.model_id)
        else:
            raise ValueError(
                f"Unsupported SigLIP2 checkpoint type '{model_type}' for {self.model_id}"
            )

        config.output_attentions = True
        return config

    def _load_model(self) -> nn.Module:
        """Load a compatible SigLIP vision encoder from HuggingFace."""
        config = self._load_config()
        if isinstance(config, Siglip2VisionConfig):
            return Siglip2VisionModel.from_pretrained(self.model_id, config=config)
        return SiglipVisionModel.from_pretrained(self.model_id, config=config)

    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Extract standardized output from SigLIP 2 vision output.

        SigLIP 2 vision encoder returns:
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

        cls_token = pooler_output  # (B, 768)
        patch_tokens = last_hidden  # (B, 196, 768)
        attention_weights = list(attentions)

        all_hidden_states = None
        if include_hidden_states and model_output.hidden_states is not None:
            all_hidden_states = list(model_output.hidden_states[1:])

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attention_weights,
            hidden_states=all_hidden_states,
        )


def create_siglip2(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SigLIP2:
    """Create a SigLIP 2 vision encoder instance."""
    return SigLIP2(device=device, dtype=dtype)
