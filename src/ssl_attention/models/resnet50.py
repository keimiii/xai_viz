"""ResNet-50 supervised baseline for SSL comparison.

ResNet-50 is a CNN pretrained on ImageNet with supervised learning. Unlike
Vision Transformers, it has no self-attention mechanism. We use Grad-CAM
to generate attention-like heatmaps via gradients.

Architectural differences from ViTs:
- 4 convolutional stages (layer1-4) instead of 12 transformer layers
- Global average pooling instead of CLS token
- 7x7 final feature grid (49 positions) instead of patch tokens

Reference:
    He et al. (2016), "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

if TYPE_CHECKING:
    from torchvision.models import ResNet

from ssl_attention.config import DEFAULT_IMAGE_SIZE, EPSILON, MODELS
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

# Load configuration from central config
_config = MODELS["resnet50"]


class ResNet50Processor:
    """Preprocessor compatible with BaseVisionModel.preprocess() interface.

    Wraps torchvision transforms to match the HuggingFace processor API.
    """

    def __init__(self) -> None:
        # Use ImageNet normalization (same as ResNet50_Weights.IMAGENET1K_V2)
        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.size = {"height": DEFAULT_IMAGE_SIZE, "width": DEFAULT_IMAGE_SIZE}

    def __call__(
        self, images: list[Image.Image], return_tensors: str = "pt"
    ) -> dict[str, Tensor]:
        """Process images like a HuggingFace processor.

        Args:
            images: List of PIL Images.
            return_tensors: Tensor format (only "pt" supported).

        Returns:
            Dict with "pixel_values" tensor.
        """
        tensors = [self.transform(img) for img in images]
        pixel_values = torch.stack(tensors, dim=0)
        return {"pixel_values": pixel_values}


class ResNet50(BaseVisionModel):
    """ResNet-50 wrapper for supervised baseline comparison.

    Uses Grad-CAM to compute heatmaps from the final convolutional layer.
    The heatmaps are returned as "attention_weights" for compatibility
    with the ViT-based attention visualization pipeline.

    Unlike ViTs, ResNet has 4 stages (layer1-4) instead of 12 layers.
    The layer slider in the frontend will automatically adapt.

    Example:
        >>> model = ResNet50()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.attention_weights[3].shape)  # (1, 224, 224) - layer 3 heatmap
    """

    model_name = "resnet50"
    model_id = _config.model_id
    patch_size = _config.patch_size
    embed_dim = _config.embed_dim
    num_layers = _config.num_layers
    num_heads = _config.num_heads
    num_registers = _config.num_registers

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize ResNet-50 with Grad-CAM hooks.

        Args:
            device: Target device. Auto-detects if None.
            dtype: Tensor dtype. Uses optimal for device if None.
        """
        # Storage for Grad-CAM
        self._activations: dict[str, Tensor] = {}
        self._gradients: dict[str, Tensor] = {}
        self._hooks: list[Any] = []

        super().__init__(device=device, dtype=dtype)

        # Register hooks for all 4 stages
        self._register_gradcam_hooks()

    def _load_processor(self) -> ResNet50Processor:
        """Load custom processor for ResNet.

        Returns:
            ResNet50Processor with ImageNet normalization.
        """
        return ResNet50Processor()

    def _load_model(self) -> nn.Module:
        """Load ResNet-50 with ImageNet weights.

        Returns:
            ResNet-50 model with pretrained weights.
        """
        model: nn.Module = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze parameters to avoid computing parameter gradients during Grad-CAM backward
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _register_gradcam_hooks(self) -> None:
        """Register forward/backward hooks on all ResNet stages."""
        # Cast to ResNet for attribute access (base class types as nn.Module)
        resnet: ResNet = self.model  # type: ignore[assignment,unused-ignore]
        target_layers: list[tuple[str, nn.Module]] = [
            ("layer1", resnet.layer1),
            ("layer2", resnet.layer2),
            ("layer3", resnet.layer3),
            ("layer4", resnet.layer4),
        ]

        for name, layer in target_layers:
            # Forward hook: capture activations
            def make_forward_hook(
                layer_name: str,
            ) -> Callable[[nn.Module, Any, Tensor], None]:
                def hook(module: nn.Module, input: Any, output: Tensor) -> None:
                    self._activations[layer_name] = output.detach()
                return hook

            # Backward hook: capture gradients
            def make_backward_hook(
                layer_name: str,
            ) -> Callable[
                [nn.Module, tuple[Tensor, ...] | Tensor, tuple[Tensor, ...] | Tensor],
                tuple[Tensor, ...] | Tensor | None,
            ]:
                def hook(
                    module: nn.Module,
                    grad_in: tuple[Tensor, ...] | Tensor,
                    grad_out: tuple[Tensor, ...] | Tensor,
                ) -> None:
                    # grad_out is a tuple; first element is the gradient we want
                    if isinstance(grad_out, tuple):
                        self._gradients[layer_name] = grad_out[0].detach()
                    else:
                        self._gradients[layer_name] = grad_out.detach()
                return hook

            self._hooks.append(layer.register_forward_hook(make_forward_hook(name)))
            self._hooks.append(layer.register_full_backward_hook(make_backward_hook(name)))

    def _remove_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        """Cleanup hooks on deletion."""
        self._remove_hooks()

    @contextmanager
    def inference_context(self) -> Generator[None, None, None]:
        """Context manager for inference WITH gradients (needed for Grad-CAM).

        Unlike ViTs, we need gradients to compute Grad-CAM heatmaps.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            # No torch.no_grad() - we need gradients for Grad-CAM!
            yield
        finally:
            if was_training:
                self.model.train()
            # Clear stored activations/gradients
            self._activations.clear()
            self._gradients.clear()

    def _compute_gradcam_heatmap(self, layer_name: str, image_size: int) -> Tensor:
        """Compute Grad-CAM heatmap for a specific layer.

        Args:
            layer_name: Name of the layer (e.g., "layer4").
            image_size: Target image size for upsampling.

        Returns:
            Heatmap tensor of shape (B, H, W).
        """
        activations = self._activations[layer_name]  # (B, C, H, W)
        gradients = self._gradients[layer_name]  # (B, C, H, W)

        # Global average pooling of gradients -> channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)  # (B, H, W)

        # ReLU to keep only positive contributions
        cam = torch.relu(cam)

        # Upsample to image size
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),  # (B, 1, H, W)
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        # Normalize per sample
        batch_size = cam.shape[0]
        flat = cam.view(batch_size, -1)
        min_val = flat.min(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        max_val = flat.max(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        cam = (cam - min_val) / (max_val - min_val + EPSILON)

        return cam

    def forward(
        self, images: Tensor, output_hidden_states: bool = False
    ) -> ModelOutput:
        """Process images through ResNet-50 with Grad-CAM.

        Args:
            images: Preprocessed images from preprocess(), shape (B, C, H, W).
            output_hidden_states: Ignored (for API compatibility).

        Returns:
            ModelOutput with Grad-CAM heatmaps as attention_weights.
        """
        with self.inference_context():
            # Enable gradients for input (needed for backward pass)
            images = images.requires_grad_(True)

            # Forward pass
            logits = self.model(images)  # (B, 1000) - ImageNet classes

            # Clear stale parameter gradients from any previous call
            self.model.zero_grad()

            # Backward pass from predicted class (for Grad-CAM)
            # Use sum of all class logits as target (class-agnostic saliency)
            logits.sum().backward()

            # Compute heatmaps for all 4 stages
            heatmaps = []
            layer_names = ["layer1", "layer2", "layer3", "layer4"]
            image_size = images.shape[-1]

            for layer_name in layer_names:
                heatmap = self._compute_gradcam_heatmap(layer_name, image_size)
                heatmaps.append(heatmap)

            # For CNN models, we don't have CLS/patch tokens in the ViT sense
            # Use the final GAP features as a pseudo-CLS token
            # Reuse layer4 activations from Grad-CAM hooks (avoid redundant forward pass)
            layer4_activations = self._activations["layer4"]  # (B, 2048, 7, 7)
            # Apply global average pooling to match ResNet's avgpool layer
            cls_features = layer4_activations.mean(dim=(2, 3))  # (B, 2048)

            return ModelOutput(
                cls_token=cls_features,  # Global average pooled features
                patch_tokens=None,  # CNNs don't have patch tokens
                attention_weights=heatmaps,  # Grad-CAM heatmaps per stage
                hidden_states=None,
            )

    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Not used - forward() handles everything for ResNet."""
        raise NotImplementedError("ResNet50 uses custom forward()")


def create_resnet50(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ResNet50:
    """Create a ResNet-50 model instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured ResNet-50 model.
    """
    return ResNet50(device=device, dtype=dtype)
