# Q1 Continuous Baseline Comparison

Scope: frozen dashboard models, scored on MSE, KL, and EMD using each model's best default-method layer from the metrics database. Lower is better for all three metrics.

Baseline source: documented constants from `docs/reference/metrics_methodology.md`.

## Headline Findings

- MSE: siglip2, siglip, dinov2, clip, resnet50, dinov3 beat all four baselines.
- MSE: no models score worse than the random baseline.
- KL: dinov3 beat all four baselines.
- KL: no models score worse than the random baseline.
- EMD: dinov3 beat all four baselines.
- EMD: siglip, siglip2 score worse than the random baseline.
- Across MSE, KL, and EMD: dinov3 beat all four baselines on every continuous metric.

## Cross-metric Divergences

- dinov2 beats all four baselines on MSE but has a weaker distribution-level story: KL beats Random, Sobel Edge; EMD beats Random, Sobel Edge.
- dinov3 beats all four baselines on MSE, KL, and EMD at its best default-method layer.
- mae shows a cross-metric spread: MSE beats Random, Center Gaussian, Saliency Prior but emd beats random.
- clip beats all four baselines on MSE but has a weaker distribution-level story: KL beats Random, Sobel Edge; EMD beats Random.
- siglip beats all four baselines on MSE but has a weaker distribution-level story: KL beats Random, Sobel Edge; EMD beats no baselines.
- siglip falls below the random baseline on EMD even though its strongest metric is MSE.
- siglip2 beats all four baselines on MSE but has a weaker distribution-level story: KL beats Random, Sobel Edge; EMD beats no baselines.
- siglip2 falls below the random baseline on EMD even though its strongest metric is MSE.
- resnet50 beats all four baselines on MSE but has a weaker distribution-level story: KL beats Random, Sobel Edge; EMD beats Random, Sobel Edge.

## Per-metric Comparison

### MSE

Baseline references: Random 0.3192, Center Gaussian 0.1770, Saliency Prior 0.0957, Sobel Edge 0.0376

| Rank | Model | Score | Best layer | Method | Beats |
| --- | --- | --- | --- | --- | --- |
| 1 | siglip2 | 0.0175 | layer6 | mean | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 2 | siglip | 0.0175 | layer6 | mean | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 3 | dinov2 | 0.0209 | layer0 | cls | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 4 | clip | 0.0211 | layer6 | cls | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 5 | resnet50 | 0.0242 | layer2 | gradcam | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 6 | dinov3 | 0.0270 | layer0 | cls | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 7 | mae | 0.0483 | layer3 | cls | Random, Center Gaussian, Saliency Prior |

### KL

Baseline references: Random 3.3627, Center Gaussian 2.6317, Saliency Prior 2.6111, Sobel Edge 3.2237

| Rank | Model | Score | Best layer | Method | Beats |
| --- | --- | --- | --- | --- | --- |
| 1 | dinov3 | 2.3247 | layer11 | cls | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 2 | dinov2 | 2.6842 | layer11 | cls | Random, Sobel Edge |
| 3 | resnet50 | 2.6917 | layer3 | gradcam | Random, Sobel Edge |
| 4 | mae | 2.7562 | layer10 | cls | Random, Sobel Edge |
| 5 | clip | 2.9122 | layer0 | cls | Random, Sobel Edge |
| 6 | siglip | 3.0020 | layer4 | mean | Random, Sobel Edge |
| 7 | siglip2 | 3.0710 | layer4 | mean | Random, Sobel Edge |

### EMD

Baseline references: Random 0.3468, Center Gaussian 0.2836, Saliency Prior 0.2654, Sobel Edge 0.3137

| Rank | Model | Score | Best layer | Method | Beats |
| --- | --- | --- | --- | --- | --- |
| 1 | dinov3 | 0.2600 | layer11 | cls | Random, Center Gaussian, Saliency Prior, Sobel Edge |
| 2 | dinov2 | 0.2978 | layer11 | cls | Random, Sobel Edge |
| 3 | resnet50 | 0.3025 | layer3 | gradcam | Random, Sobel Edge |
| 4 | mae | 0.3177 | layer10 | cls | Random |
| 5 | clip | 0.3261 | layer0 | cls | Random |
| 6 | siglip | 0.3476 | layer4 | mean | None |
| 7 | siglip2 | 0.3538 | layer4 | mean | None |

## Per-model Wrap-up

- dinov2: MSE beats all four baselines; KL beats Random, Sobel Edge; EMD beats Random, Sobel Edge.
- dinov3: MSE beats all four baselines; KL beats all four baselines; EMD beats all four baselines.
- mae: MSE beats Random, Center Gaussian, Saliency Prior; KL beats Random, Sobel Edge; EMD beats Random.
- clip: MSE beats all four baselines; KL beats Random, Sobel Edge; EMD beats Random.
- siglip: MSE beats all four baselines; KL beats Random, Sobel Edge; EMD beats no baselines.
- siglip2: MSE beats all four baselines; KL beats Random, Sobel Edge; EMD beats no baselines.
- resnet50: MSE beats all four baselines; KL beats Random, Sobel Edge; EMD beats Random, Sobel Edge.
