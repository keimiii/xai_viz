"""Canonical model order, colours, and display names for final-report figures.

These constants are the single source of truth for how the six frozen models are
ordered and coloured across the report. The order matches Tables 4 and 6 and
Figures 5 and 6; the colours reuse the dashboard's layer-progression palette so
the report figures match the live UI. The dashboard colours each line with a
golden-angle HSL sweep, `hsl((i * 137.5) % 360, 70%, 50%)` (see
`app/frontend/src/pages/Dashboard.tsx`); the hex values below are indices 0..5 of
that sweep, assigned in MODEL_ORDER. Import via
`from experiments.scripts.figures._palette import MODEL_ORDER, MODEL_COLORS` to
stay consistent across figures.

ResNet-50 is intentionally excluded: it is a CNN baseline with no transformer
layers and is not part of the per-layer transformer comparison.
"""

from __future__ import annotations

# Canonical model order (internal DB / config keys). Must match Tables 4 & 6
# and Figures 5 & 6.
MODEL_ORDER: tuple[str, ...] = (
    "dinov2",
    "dinov3",
    "mae",
    "clip",
    "siglip",
    "siglip2",
)

# Dashboard golden-angle HSL palette (hsl((i*137.5)%360, 70%, 50%)), indices
# 0..5, assigned in MODEL_ORDER so report figures match the live UI.
MODEL_COLORS: dict[str, str] = {
    "dinov2": "#D82626",   # red    (i=0)
    "dinov3": "#26D85A",   # green  (i=1)
    "mae": "#8E26D8",      # purple (i=2)
    "clip": "#D8C226",     # gold   (i=3)
    "siglip": "#26BAD8",   # cyan   (i=4)
    "siglip2": "#D82686",  # magenta (i=5)
}

# Human-readable labels for legends and captions.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "mae": "MAE",
    "clip": "CLIP",
    "siglip": "SigLIP",
    "siglip2": "SigLIP2",
}
