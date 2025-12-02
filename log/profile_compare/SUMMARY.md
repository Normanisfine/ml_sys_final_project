# Quick Summary: 3DGS vs TC-GS

## Performance (Garden Scene, 1036×1600)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME TIME BREAKDOWN (GPU)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original 3DGS (~19.8 ms per frame, ~35 FPS)                   │
│  ├── Preprocessing ████░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.66 ms    │
│  ├── TileBinning   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.28 ms    │
│  ├── Sorting       ██████░░░░░░░░░░░░░░░░░░░░░░░░░  4.28 ms ⚠️ │
│  └── AlphaBlending ████████████████████░░░░░░░░░░░ 11.62 ms ⚠️ │
│                                                                 │
│  TC-GS (~10.4 ms per frame, ~124 FPS) — 1.9× FASTER            │
│  ├── Preprocessing ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.17 ms ✓  │
│  ├── TileBinning   █████░░░░░░░░░░░░░░░░░░░░░░░░░░  2.94 ms    │
│  ├── Sorting       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.12 ms ✓  │
│  └── AlphaBlending ██████████████░░░░░░░░░░░░░░░░░  6.18 ms ✓  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Per-Stage Comparison

| Stage | Original 3DGS | TC-GS | Change |
|-------|--------------|-------|--------|
| Preprocessing | 2.66 ms | 1.17 ms | **2.3× faster** |
| TileBinning | 1.28 ms | 2.94 ms | 2.3× slower (trade-off) |
| Sorting | 4.28 ms | 0.12 ms | **35× faster** ⭐ |
| AlphaBlending | 11.62 ms | 6.18 ms | **1.9× faster** |
| **Total** | **19.8 ms** | **10.4 ms** | **1.9× faster** |

## Why TC-GS is Faster

| Stage | What Changed | Effect |
|-------|--------------|--------|
| Preprocessing | Tighter culling, fewer Gaussians pass | 2.3× faster |
| TileBinning | SnugBox ellipse intersection (more work) | 2.3× slower |
| **Sorting** | **~30-50% fewer tile-Gaussian pairs** | **35× faster** |
| AlphaBlending | Tensor Cores + FP16 + batching | 1.9× faster |

**Key Insight**: TC-GS trades off slower TileBinning for **dramatically faster Sorting and Blending**.

## Core Innovation

```
Original 3DGS:                    TC-GS:
┌─────────────────────┐           ┌─────────────────────┐
│ For each Gaussian:  │           │ Batch 16 Gaussians: │
│   power = ...       │           │   U = pixel_matrix  │
│   alpha = exp(power)│    →      │   V = gauss_matrix  │
│   color += ...      │           │   B = U^T × V (MMA) │
│ (CUDA cores, FP32)  │           │ (Tensor cores, FP16)│
└─────────────────────┘           └─────────────────────┘
```

## Quality Impact

| Metric | 3DGS | TC-GS | Δ |
|--------|------|-------|---|
| PSNR (train) | 29.75 | 29.73 | -0.02 dB |
| PSNR (test) | 27.38 | 27.37 | -0.01 dB |

**Conclusion**: Negligible quality loss due to local coordinate transformation mitigating FP16 precision issues.

---

*See `PROFILE_COMPARISON.md` for detailed analysis.*

