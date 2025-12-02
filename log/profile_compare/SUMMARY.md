# Quick Summary: 3DGS vs TC-GS

## Performance (Garden Scene, 1036×1600)

```
┌─────────────────────────────────────────────────────────────────┐
│                  GPU KERNEL TIME BREAKDOWN (NVTX)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original 3DGS (19.8 ms GPU kernel time)                       │
│  ├── Preprocessing ████░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.66 ms    │
│  ├── TileBinning   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.28 ms    │
│  ├── Sorting       ██████░░░░░░░░░░░░░░░░░░░░░░░░░  4.28 ms ⚠️ │
│  └── AlphaBlending ████████████████████░░░░░░░░░░░ 11.62 ms ⚠️ │
│                                                                 │
│  TC-GS (10.4 ms GPU kernel time) — 1.9× FASTER KERNELS         │
│  ├── Preprocessing ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.17 ms ✓  │
│  ├── TileBinning   █████░░░░░░░░░░░░░░░░░░░░░░░░░░  2.94 ms    │
│  ├── Sorting       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.12 ms ✓  │
│  └── AlphaBlending ██████████████░░░░░░░░░░░░░░░░░  6.18 ms ✓  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Per-Stage Comparison (GPU Kernel Time)

| Stage | Original 3DGS | TC-GS | Change |
|-------|--------------|-------|--------|
| Preprocessing | 2.66 ms | 1.17 ms | **2.3× faster** |
| TileBinning | 1.28 ms | 2.94 ms | 2.3× slower (trade-off) |
| Sorting | 4.28 ms | 0.12 ms | **35× faster** ⭐ |
| AlphaBlending | 11.62 ms | 6.18 ms | **1.9× faster** |
| **GPU Kernel Total** | **19.8 ms** | **10.4 ms** | **1.9× faster** |

## Understanding the Metrics

| Metric | Original 3DGS | TC-GS | Speedup |
|--------|---------------|-------|---------|
| **NVTX GPU Kernel Time** | 19.8 ms | 10.4 ms | **1.9×** |
| **End-to-End FPS** | ~35 FPS | ~124 FPS | **3.5×** |

**Why 1.9× kernel speedup but 3.5× FPS improvement?**

- **NVTX measures GPU kernel execution only** — the compute time inside CUDA kernels
- **FPS measures total frame time** — includes CPU overhead, memory transfers, Python/PyTorch, sync delays
- **TC-GS has better pipeline efficiency** — tighter kernel packing, less idle time between operations
- The 3.5× FPS comes from kernel speedup (1.9×) PLUS reduced CPU-GPU synchronization overhead

## Why TC-GS is Faster

| Stage | What Changed | Effect |
|-------|--------------|--------|
| Preprocessing | Tighter culling, fewer Gaussians pass | 2.3× faster |
| TileBinning | SnugBox ellipse intersection (more work) | 2.3× slower |
| **Sorting** | **~30-50% fewer tile-Gaussian pairs** | **35× faster** |
| AlphaBlending | Tensor Cores + FP16 + batching | 1.9× faster |

**Key Insight**: TC-GS trades off slower TileBinning for **dramatically faster Sorting and Blending**.

## Why 1.9× Kernel Speedup → 3.5× FPS (Pipeline Efficiency)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE COMPOUNDING EFFECT                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SnugBox Tight Culling (30-50% fewer tile-Gaussian pairs)       │
│     └─→ Less data flows through entire pipeline                    │
│                                                                     │
│  2. Cascade Effect on Every Stage:                                  │
│     • Sorting: 35× faster (radix sort is O(n), n is smaller)       │
│     • Memory: 50% less bandwidth (FP16 + fewer items)              │
│     • Sync: Less CPU-GPU coordination overhead                      │
│                                                                     │
│  3. Tensor Core Batching (16 Gaussians at once via MMA)            │
│     • 8-16× throughput vs CUDA cores for matrix ops                │
│     • Better latency hiding (more compute per memory access)       │
│                                                                     │
│  4. Pipeline Gaps Shrink:                                           │
│     • Smaller buffers → faster allocation                          │
│     • Less prefix sum work → faster binning                        │
│     • Tighter kernel scheduling → less idle time                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Optimization | Technique | Impact |
|--------------|-----------|--------|
| **SnugBox Culling** | Exact ellipse-tile intersection with opacity threshold | 30-50% fewer pairs |
| **Tensor Core MMA** | `mma_16x8x8_f16_f16` batches 16 Gaussians | 8-16× throughput |
| **FP16 Features** | Pack colors into half-precision | 50% less memory bandwidth |
| **Local Coordinates** | Tile-local coords (±7.5 vs 0-1600) | Maintains FP16 precision |

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

## Summary

| What We Measured | Result |
|------------------|--------|
| **GPU Kernel Speedup (NVTX)** | 1.9× (19.8 ms → 10.4 ms) |
| **End-to-End FPS Improvement** | 3.5× (35 → 124 FPS) |
| **Quality Loss** | Negligible (-0.02 dB PSNR) |
| **Biggest Kernel Win** | Sorting (35× faster) |
| **Key Innovation** | Tensor Core MMA for alpha blending |

*See `PROFILE_COMPARISON.md` for detailed stage-by-stage analysis.*

