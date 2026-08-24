# Architecture Notes

## Design goals

The model is designed to combine complementary inductive biases:

- CNN features preserve local texture and anatomical edges.
- ViT features model long-range relationships across the image.
- H-CSAF learns how much global context should influence each CNN scale.
- The edge branch explicitly supervises boundary-sensitive representations.
- The decoder reconstructs high-resolution segmentation maps with skip connections.
- Deep supervision improves optimization of intermediate decoder representations.

## Attention implementation

Where supported by the installed PyTorch version, scaled dot-product attention can dispatch to optimized attention kernels. The architecture keeps attention isolated in fusion modules so attention implementations can be benchmarked independently.

## Ablation plan

Recommended experiments:

1. CNN-only baseline
2. CNN + ViT without H-CSAF
3. CNN + ViT + concatenation fusion
4. CNN + ViT + H-CSAF
5. H-CSAF without edge branch
6. H-CSAF + edge branch
7. Full model without deep supervision
8. Full model with deep supervision
