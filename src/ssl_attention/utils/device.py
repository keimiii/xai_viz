"""Device detection and memory management utilities."""

import gc

import torch


def get_device() -> torch.device:
    """Auto-detect the best available device: CUDA > MPS > CPU.

    Returns:
        torch.device: The best available compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype_for_device(device: torch.device | None = None) -> torch.dtype:
    """Get the optimal dtype for a device.

    - CUDA: bfloat16 for better performance (if supported)
    - MPS: float32 (bfloat16 has limited support)
    - CPU: float32

    Args:
        device: Target device. If None, auto-detects.

    Returns:
        torch.dtype: Optimal dtype for the device.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        # Check for bfloat16 support (Ampere+ GPUs)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # MPS and CPU use float32
    return torch.float32


def clear_memory(device: torch.device | None = None) -> None:
    """Clear device memory cache.

    Important for MPS which doesn't automatically free memory.

    Args:
        device: Device to clear. If None, clears all available.
    """
    gc.collect()

    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        # MPS requires explicit cache clearing
        torch.mps.empty_cache()
        torch.mps.synchronize()


def device_info() -> dict[str, str | int | float]:
    """Get information about the current device.

    Returns:
        dict with device type, name, and memory info if available.
    """
    device = get_device()
    info: dict[str, str | int | float] = {"device": str(device)}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
        # MPS doesn't expose memory info directly
    else:
        info["name"] = "CPU"

    return info
