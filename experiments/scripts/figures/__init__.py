"""Shared building blocks for final-report figures.

Currently exposes the canonical model order and colour-blind-safe palette so
every report figure draws the six frozen models in the same order with the same
colours. Import from here rather than redefining per-script:

    from experiments.scripts.figures._palette import MODEL_ORDER, MODEL_COLORS
"""

from experiments.scripts.figures._palette import (
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
)

__all__ = ["MODEL_ORDER", "MODEL_COLORS", "MODEL_DISPLAY_NAMES"]
