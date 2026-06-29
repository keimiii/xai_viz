"""Pre-computation scripts for the visualization app.

These scripts pre-compute all attention maps, heatmap images, and metrics
before the app runs, avoiding expensive model inference at request time.

Run in order:
1. generate_attention_cache.py - Extract attention from all models/layers
2. generate_heatmap_images.py - Render PNG overlays for fast serving
3. generate_metrics_cache.py - Compute IoU at multiple percentiles
"""
