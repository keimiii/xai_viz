/**
 * Glossary of technical terms for educational tooltips.
 * Keys should match labels used in the UI for easy lookup.
 */

export const GLOSSARY: Record<string, string> = {
  'Attention Head':
    'Individual transformer attention head. "All (Fused)" averages all heads, while a specific head lets you inspect one learned attention pattern at a time.',
  'Attention Method':
    'How attention is computed. CLS uses class token attention. Rollout accumulates attention across layers.',
  'Attention Threshold':
    'Filters to show only top-attended regions. "Top 10%" shows patches in the highest 10% of attention values.',
  'Similarity Heatmap':
    'Shows how similar each region is to the selected bounding box based on learned features.',
  'Heatmap Opacity':
    'Controls transparency of the active overlay. It affects the attention heatmap in Head Attention mode and the similarity overlay in Feature Similarity mode.',
  'Heatmap Style':
    'Visual style for the active overlay. Smooth uses interpolation, while Squares and Circles emphasize discrete patch values.',
  Layer:
    'Network depth. Early layers capture edges/textures. Later layers capture semantic concepts. Layer count varies by model.',
  Model: 'Vision model for feature extraction. Each has different architecture and training.',
  'Show Bounding Boxes': 'Toggle visibility of annotated bounding boxes on the image.',
  'IoU Score':
    'Overlap between thresholded attention and the annotation. '
    + 'Higher is better. Changes when percentile changes. '
    + '\nUse it to judge how tightly the highlighted region lines up with the labeled feature.',
  Coverage:
    'Fraction of attention mass inside the annotation. '
    + 'Higher is better. Threshold-free for a fixed image/model/method. '
    + '\nUse it to see whether the model is spending its attention on the feature rather than the background.',
  MSE:
    'Mean squared error against the Gaussian soft-union target. '
    + 'Lower is better. Threshold-free for a fixed image/model/method. '
    + '\nUse it to judge whether the overall attention shape matches the annotated feature, not just the thresholded overlap.',
  KL:
    'KL divergence using KL(GT || attention) after both heatmaps are converted into smoothed probability distributions. '
    + 'Lower is better. Threshold-free for a fixed image/model/method. '
    + '\nUse it to judge how much probability mass the model misses or spreads away from the Gaussian ground-truth target.',
  EMD:
    'Earth Mover\'s Distance (Wasserstein-1) on a shared 8x8 support after both heatmaps are resized and normalized into probability distributions. '
    + 'Lower is better. Threshold-free for a fixed image/model/method. '
    + '\nUse it to judge how far the attention mass would need to move spatially to match the Gaussian ground-truth target.',
};
