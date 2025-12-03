# Optimize Tensor-Core-Based Gaussian Splatting CUDA Kernel

[![CUDA](https://img.shields.io/badge/CUDA-11.6+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

## Overview

This project investigates and profiles the performance optimizations introduced by **TC-GS (Tensor-Core Gaussian Splatting)** compared to the original **3D Gaussian Splatting** implementation.

Almost all existing Gaussian Splatting systems use CUDA cores only for computation. The TC-GS paper is the **first to leverage Tensor Cores** to accelerate Gaussian Splatting's rendering kernel. This project profiles both implementations to identify performance characteristics and bottlenecks.

### Key Finding

TC-GS achieves **~3.5× speedup** (35 FPS → 124 FPS) with negligible quality loss (~0.02 dB PSNR) on the Garden scene at 1036×1600 resolution.

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

---

## Project Structure

```
final_project/
├── README.md                    # This file
├── 2505.24796v2.pdf             # TC-GS Paper
│
├── gs_profile/                  # Original 3D Gaussian Splatting (submodule)
│   ├── submodules/
│   │   └── diff-gaussian-rasterization/
│   │       ├── cuda_rasterizer/
│   │       │   └── forward.cu   # CUDA core-based rendering kernel
│   │       └── NVTX_MARKERS.md  # Profiling marker documentation
│   └── render_fps.py            # FPS benchmark script
│
├── tcgs_profile/                # TC-GS Implementation (submodule)
│   ├── CHANGELOG.md             # Compatibility fixes (CUDA 11.6)
│   ├── submodules/
│   │   └── tcgs_speedy_rasterizer/
│   │       ├── cuda_rasterizer/
│   │       │   ├── forward.cu   # Tensor-core rendering kernel
│   │       │   └── tcgs/        # TC-GS specific CUDA code
│   │       │       ├── tcgs_forward.cu
│   │       │       └── tcgs_utils.h
│   │       └── NVTX_MARKERS.md  # Profiling marker documentation
│   └── render.py                # Rendering script
│
├── log/                         # Profiling logs and analysis
│   ├── env.sh                   # Environment setup script
│   ├── profile.sh               # nsys profiling commands
│   ├── profile.log              # Profiling output log
│   ├── profile_nsys/            # Nsight Systems profiling reports
│   │   ├── gs_profile_*.nsys-rep
│   │   └── tcgs_profile_*.nsys-rep
│   └── profile_compare/         # Analysis results
│       ├── SUMMARY.md           # Quick performance summary
│       └── PROFILE_COMPARISON.md # Detailed stage-by-stage analysis
│
├── assets/                      # Pre-trained models (gitignored)
│   └── garden_3dgs/             # Garden scene checkpoint
│
└── envs/                        # Conda environments (gitignored)
    └── .conda/
        ├── gs_env/              # Environment for original 3DGS
        └── tcgs_env/            # Environment for TC-GS
```

---

## Submodule Composition

This project contains **two git submodules** for comparing the implementations:

| Submodule | Repository | Description |
|-----------|------------|-------------|
| `gs_profile` | [Normanisfine/gs_profile](https://github.com/Normanisfine/gs_profile.git) | Original 3D Gaussian Splatting with NVTX profiling markers |
| `tcgs_profile` | [Normanisfine/tcgs_profile](https://github.com/Normanisfine/tcgs_profile.git) | TC-GS implementation with NVTX profiling markers |

### Cloning with Submodules

```bash
git clone --recursive https://github.com/YOUR_USERNAME/final_project.git
# Or if already cloned:
git submodule update --init --recursive
```

---

## Key Files Reference

### 📊 Profiling & Analysis

| File | Description |
|------|-------------|
| [`log/profile_compare/SUMMARY.md`](log/profile_compare/SUMMARY.md) | **Quick summary** - Performance comparison at a glance |
| [`log/profile_compare/PROFILE_COMPARISON.md`](log/profile_compare/PROFILE_COMPARISON.md) | **Detailed analysis** - Stage-by-stage breakdown with code snippets |
| [`log/profile.sh`](log/profile.sh) | nsys profiling commands for both implementations |
| [`log/profile_nsys/`](log/profile_nsys/) | Raw `.nsys-rep` files for Nsight Systems |

### 🔧 NVTX Profiling Markers

| File | Description |
|------|-------------|
| [`gs_profile/submodules/diff-gaussian-rasterization/NVTX_MARKERS.md`](https://github.com/Normanisfine/gs_profile/blob/main/submodules/diff-gaussian-rasterization/NVTX_MARKERS.md) | NVTX markers for original 3DGS |
| [`tcgs_profile/submodules/tcgs_speedy_rasterizer/NVTX_MARKERS.md`](https://github.com/Normanisfine/tcgs_profile/blob/main/submodules/tcgs_speedy_rasterizer/NVTX_MARKERS.md) | NVTX markers for TC-GS |

### 📝 Changelogs & Fixes

| File | Description |
|------|-------------|
| [`tcgs_profile/CHANGELOG.md`](https://github.com/Normanisfine/tcgs_profile/blob/main/CHANGELOG.md) | CUDA 11.6 compatibility fix for `__hmin` intrinsic |

### ⚙️ Environment Setup

| File | Description |
|------|-------------|
| [`log/env.sh`](log/env.sh) | Complete environment setup for both implementations |

---

## Performance Comparison

### Per-Stage GPU Kernel Time (NVTX Profiling)

| Stage | Original 3DGS | TC-GS | Speedup |
|-------|--------------|-------|---------|
| Preprocessing | 2.66 ms | 1.17 ms | **2.3×** |
| TileBinning | 1.28 ms | 2.94 ms | 0.4× (trade-off) |
| Sorting | 4.28 ms | 0.12 ms | **35×** ⭐ |
| AlphaBlending | 11.62 ms | 6.18 ms | **1.9×** |
| **GPU Kernel Total** | **19.8 ms** | **10.4 ms** | **1.9×** |

### Why TC-GS is Faster

| Optimization | Technique | Effect |
|--------------|-----------|--------|
| **SnugBox Culling** | Exact ellipse-tile intersection with opacity threshold | 30-50% fewer tile-Gaussian pairs |
| **Tensor Core MMA** | `mma_16x8x8_f16_f16` batches 16 Gaussians at once | 8-16× throughput vs CUDA cores |
| **FP16 Features** | Pack colors into half-precision | 50% less memory bandwidth |
| **Local Coordinates** | Tile-local transform (±7.5 vs 0-1600) | Maintains FP16 precision |

### Why 1.9× Kernel → 3.5× FPS (Cascade Effect)

```
SnugBox Culling (30-50% fewer pairs)
    ↓
┌─────────────────────────────────────────────┐
│ • Sorting: 35× faster (less data to sort)  │
│ • Memory: 50% less bandwidth               │
│ • Sync: Less CPU-GPU coordination          │
└─────────────────────────────────────────────┘
    ↓
Tensor Core MMA + FP16 (1.9× faster blending)
    ↓
Pipeline gaps shrink → 3.5× total FPS
```

---

## Understanding the Metrics

### GPU Kernel Time vs End-to-End FPS

You may notice that the **1.9× kernel speedup** differs from the **3.5× FPS improvement**. This is expected:

| Metric | What It Measures | Original 3DGS | TC-GS | Speedup |
|--------|------------------|---------------|-------|---------|
| **NVTX Kernel Time** | Sum of GPU kernel durations only | 19.8 ms | 10.4 ms | **1.9×** |
| **End-to-End FPS** | Total wall-clock time per frame | ~35 FPS | ~124 FPS | **3.5×** |

**Why the difference?**

```
Total Frame Time = GPU Kernels + CPU Overhead + Memory Transfers + Sync Delays
                   ^^^^^^^^^^^
                   (NVTX measures this part only)
```

1. **NVTX markers measure GPU kernel execution time only** — the actual compute work on the GPU
2. **FPS includes everything** — CPU overhead, Python/PyTorch operations, memory transfers, synchronization
3. **TC-GS has better pipeline efficiency** — tighter kernel packing with less idle time between operations

### Profiling Overhead

When running with `nsys profile`, there is additional overhead from:
- **CUDA API interception** — hooking into every CUDA call to record timestamps
- **NVTX range recording** — writing marker data to buffers
- **Memory tracking** — monitoring allocations with `--cuda-memory-usage=true`
- **GPU metrics collection** — sampling hardware counters with `--gpu-metrics-device=all`

The kernel durations are accurate (from GPU hardware timers), but gaps between kernels may be inflated during profiling. FPS benchmarks should be run **without the profiler** for accurate throughput measurements

---

## Quick Start

### 1. Environment Setup

```bash
# Load modules (on HPC cluster)
module purge
module load anaconda3/2024.02
module load cuda/11.6.2

# Set up conda paths
export CONDA_PKGS_DIRS="/vast/ml8347/ml_sys/final_project/envs/.conda/pkgs"
export CONDA_ENVS_DIRS="/vast/ml8347/ml_sys/final_project/envs/.conda/envs"
```

### 2. Install Original 3DGS

```bash
cd gs_profile
conda activate /vast/ml8347/ml_sys/final_project/envs/gs_env
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn submodules/fused-ssim
```

### 3. Install TC-GS

```bash
cd tcgs_profile
conda activate /vast/ml8347/ml_sys/final_project/envs/tcgs_env
pip install submodules/tcgs_speedy_rasterizer
pip install submodules/simple-knn submodules/fused-ssim
```

### 4. Run Profiling

```bash
# Profile original 3DGS
cd gs_profile
nsys profile --trace=cuda,nvtx -o gs_profile python render_fps.py \
    -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000

# Profile TC-GS
cd tcgs_profile
nsys profile --trace=cuda,nvtx -o tcgs_profile python render.py \
    -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000
```

---

## References

1. **TC-GS Paper**: [arXiv:2505.24796v2](https://arxiv.org/abs/2505.24796) - "TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores"

2. **Original 3DGS**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023

---

## License

This project is for educational purposes as part of the ML Systems course (P6.1 Final Project).

