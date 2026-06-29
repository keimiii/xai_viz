"""Utility functions.

This module provides utility functions for:
- Device detection and management
- Memory clearing
- Dtype selection
"""

from ssl_attention.utils.device import (
    clear_memory,
    device_info,
    get_device,
    get_dtype_for_device,
)

__all__ = [
    "get_device",
    "get_dtype_for_device",
    "clear_memory",
    "device_info",
]
