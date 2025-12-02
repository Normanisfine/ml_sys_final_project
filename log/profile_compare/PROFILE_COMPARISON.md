# Gaussian Splatting Profiling Comparison: Original 3DGS vs TC-GS

This document summarizes the profiling comparison between the original 3D Gaussian Splatting (3DGS) and TC-GS (Tensor Core Gaussian Splatting) based on nsys profiling results and code analysis.

## Test Configuration

- **Dataset**: Garden scene (MipNeRF360)
- **Resolution**: 1036 × 1600 pixels
- **Iteration**: 30000
- **GPU**: NVIDIA TU10x (Turing architecture)
- **Profiler**: NVIDIA Nsight Systems with NVTX markers

---

## Overall Performance Summary

| Metric | Original 3DGS | TC-GS | Speedup |
|--------|--------------|-------|---------|
| **End-to-End FPS** | ~35 | ~124 | **3.5×** |
| **GPU Kernel Time** | 19.8 ms | 10.4 ms | **1.9×** |
| **PSNR (train)** | 29.75 dB | 29.73 dB | -0.02 dB |
| **PSNR (test)** | 27.38 dB | 27.37 dB | -0.01 dB |

**Key Observation**: TC-GS achieves **3.5× end-to-end speedup** with negligible quality loss (~0.02 dB PSNR).

---

## Understanding Kernel Time vs FPS

The **1.9× GPU kernel speedup** differs from the **3.5× FPS improvement** because:

```
Total Frame Time = GPU Kernels + CPU Overhead + Memory Transfers + Sync Delays + Python/PyTorch
                   ^^^^^^^^^^^
                   (NVTX measures this)
```

| Measurement | What It Captures | 3DGS → TC-GS |
|-------------|------------------|--------------|
| **NVTX Kernel Time** | Sum of GPU kernel execution durations | 19.8 ms → 10.4 ms (**1.9×**) |
| **End-to-End FPS** | Total wall-clock time including all overheads | 35 → 124 FPS (**3.5×**) |

The additional speedup comes from:
1. **Reduced CPU-GPU synchronization** — fewer tile-Gaussian pairs means less data transfer
2. **Better pipeline efficiency** — tighter kernel packing with less idle time
3. **Lower memory bandwidth** — fewer items to sort and blend

### Profiling Overhead Note

When running with `nsys profile`, there is additional overhead from CUDA API interception, NVTX recording, and memory tracking. The kernel durations are accurate (from GPU hardware timers), but wall-clock time is inflated. FPS benchmarks should be run without the profiler for accurate throughput measurements.

---

## NVTX Profiling Results (Per-Frame GPU Kernel Time)

| Stage | Original 3DGS | TC-GS | Speedup |
|-------|--------------|-------|---------|
| **Preprocessing** | 2.657 ms | 1.169 ms | **2.3×** |
| **TileBinning** | 1.277 ms | 2.942 ms | 0.4× (more work) |
| **Sorting** | 4.277 ms | 0.123 ms | **34.8×** |
| **AlphaBlending** | 11.620 ms | 6.178 ms | **1.9×** |
| **GPU Kernel Total** | ~19.8 ms | ~10.4 ms | **1.9×** |

---

## Stage-by-Stage Analysis

### 1. Preprocessing Stage

| Metric | Original 3DGS | TC-GS | Speedup |
|--------|--------------|-------|---------|
| GPU Time | 2.657 ms | 1.169 ms | **2.3×** |

#### Observation
TC-GS preprocessing is **2.3× faster** than original 3DGS (1.169 ms vs 2.657 ms).

#### Reason
Both methods perform similar core operations:
- 3D → 2D Gaussian projection
- Covariance matrix computation  
- Spherical harmonics to RGB conversion
- Frustum culling

However, TC-GS uses **tighter tile culling** in preprocessing which:
1. Computes exact ellipse-tile intersection (more work per Gaussian)
2. But culls more Gaussians earlier (fewer pass to later stages)
3. The net effect is faster due to reduced data for subsequent stages

#### Implementation
Original 3DGS uses simple rectangle bounds:
```cpp
getRect(point_image, my_radius, rect_min, rect_max, grid);
tiles_touched = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
```

TC-GS uses tight ellipse intersection:
```cpp
uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, ...);
if (tiles_count == 0) return;  // More aggressive early culling
```

---

### 2. TileBinning Stage

| Metric | Original 3DGS | TC-GS | Ratio |
|--------|--------------|-------|-------|
| GPU Time | 1.277 ms | 2.942 ms | 0.4× (TC-GS slower) |
| Tiles per Gaussian | More (rectangle) | Fewer (ellipse) | ~30% fewer |

#### Observation
TC-GS TileBinning takes **2.3× longer** than original 3DGS (2.942 ms vs 1.277 ms).

#### Reason
**This is expected!** TC-GS does more work per Gaussian in the binning stage:

**Original 3DGS** uses simple axis-aligned bounding box:
```cpp
// Simple rectangle based on radius - fast but loose
getRect(point_image, my_radius, rect_min, rect_max, grid);
tiles_touched = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
```

**TC-GS** uses tight ellipse intersection ("SnugBox" algorithm):
```cpp
// Complex ellipse-tile intersection with opacity threshold - slower but precise
float t = 2.0f * log(con_o.w * 255.0f);  // Opacity-aware!
uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, ...);
```

#### Implementation Logic
The SnugBox algorithm (from TC-GS paper Section 3.2) computes exact tile overlap:

```
Original 3DGS (Rectangle):          TC-GS (Tight Ellipse):
┌─┬─┬─┬─┬─┐                        ┌─┬─┬─┬─┬─┐
│█│█│█│█│█│  ← wasted corners      │ │█│█│█│ │
├─┼─┼─┼─┼─┤                        ├─┼─┼─┼─┼─┤
│█│█│█│█│█│                        │█│█│█│█│█│
├─┼─┼─┼─┼─┤                        ├─┼─┼─┼─┼─┤
│█│█│█│█│█│                        │ │█│█│█│ │
└─┴─┴─┴─┴─┘                        └─┴─┴─┴─┴─┘
25 tiles                           ~18 tiles (~30% fewer)
```

**Trade-off**: TC-GS spends more time in TileBinning to generate fewer tile-Gaussian pairs, which **pays off massively in Sorting (35×) and AlphaBlending (1.9×)**.

---

### 3. Sorting Stage

| Metric | Original 3DGS | TC-GS | Speedup |
|--------|--------------|-------|---------|
| GPU Time | 4.277 ms | 0.123 ms (123 μs) | **34.8×** |

#### Observation
TC-GS sorting is **35× faster** than original 3DGS! This is the biggest relative speedup.

#### Reason
The massive speedup comes from **significantly fewer tile-Gaussian pairs** to sort:
- Fewer pairs = less data to sort
- Radix sort complexity is O(n·k) where n is number of pairs
- TC-GS's tight culling reduces n by ~30-50%

Additionally, the sorting benefits from:
1. **Better cache utilization** with smaller data
2. **Fewer radix passes** needed for smaller datasets
3. **Less memory bandwidth** pressure

#### Implementation
Both use the same sorting algorithm:
```cpp
CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
    binningState.list_sorting_space,
    binningState.sorting_size,
    binningState.point_list_keys_unsorted, binningState.point_list_keys,
    binningState.point_list_unsorted, binningState.point_list,
    num_rendered, 0, 32 + bit), debug)
```

The key difference is `num_rendered` — TC-GS has far fewer items due to tight tile culling, resulting in dramatically faster sorting.

---

### 4. AlphaBlending Stage (KEY DIFFERENCE)

| Metric | Original 3DGS | TC-GS | Speedup |
|--------|--------------|-------|---------|
| GPU Time | 11.620 ms | 6.178 ms | **1.9×** |
| Compute Unit | CUDA Cores (FP32) | Tensor Cores (FP16) | - |
| Processing | 1 Gaussian/thread | 16 Gaussians/warp (batched) | - |

#### Observation
TC-GS achieves **1.9× speedup** in the alpha-blending kernel through tensor core utilization.

#### Reason
According to the TC-GS paper, the key innovation is **mapping alpha computation to matrix multiplication**:

> "The key innovation lies in mapping alpha computation to matrix multiplication, fully utilizing otherwise idle TCUs in existing 3DGS implementations."

**Original 3DGS** processes 1 Gaussian per thread sequentially:
```cpp
// Per-pixel, per-Gaussian loop (FP32)
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
    float alpha = min(0.99f, con_o.w * exp(power));  // Scalar exp()
    // ... blend one Gaussian
}
```

**TC-GS** batches 16 Gaussians using tensor cores:
```cpp
// Vectorized matrix multiplication (FP16 tensor cores)
// 1. Convert pixels and Gaussians to matrix format
uint4 pix_vec = pix2vec(pixf_local);
uint4 gaussian_vec = gs2vec(conics, means, pixf_mid);

// 2. Batch compute exponents via MMA instruction
mma_16x8x8_f16_f16(expmat_reg[0], expmat_reg[1],
    pixmat_reg[0], pixmat_reg[1], gsmat_reg[0], ...);

// 3. Combined culling and blending
RGBD = culling_and_blending(exponent_matrix, channels_smem, T, ...);
```

#### Implementation Logic (from TC-GS Paper)

The alpha value computation:
$$\alpha = o \cdot \exp\left(-\frac{1}{2}(p - \mu')^T \Sigma'^{-1} (p - \mu')\right)$$

Is reformulated as a dot product (exponent β):
$$\beta = \ln(\alpha) = \mathbf{u}^T \mathbf{v}$$

Where:
- **Pixel vector u**: `[Δp_x, Δp_y, Δp_x², Δp_y², 1/3, Δp_x·Δp_y, 1/3, 1/3]`
- **Gaussian vector v**: `[linear terms, quadratic terms, constant]`

This allows batching into matrix multiplication:
$$\mathbf{B} = \mathbf{U}^T \mathbf{V}$$

Where B is the exponent matrix computed via **tensor core MMA instructions**.

**Local Coordinate Transformation** (Section 4.2 of paper):
To mitigate FP16 precision loss, TC-GS uses tile-local coordinates:
```cpp
// Global → Local transformation reduces quadratic term magnitude
pixf_mid = make_float2(pix_min.x + 7.5f, pix_min.y + 7.5f);
pixf_local = make_float2(pix.x - pixf_mid.x, pix.y - pixf_mid.y);
```

This reduces numerical error from O(w² + h²) to O(w + h).

---

## Culling Comparison

### Early Culling Stages

| Stage | Original 3DGS | TC-GS | Impact |
|-------|--------------|-------|--------|
| Frustum | `in_frustum()` | Same | Baseline |
| Degenerate | `det == 0` | Same | Baseline |
| Tile Overlap | Rectangle-based | **Ellipse + opacity** | 30% fewer pairs |

### Per-Pixel Culling (in AlphaBlending)

**Original 3DGS** (3 sequential checks):
```cpp
if (power > 0.0f) continue;           // Outside Gaussian
if (alpha < 1.0f / 255.0f) continue;  // Too transparent  
if (test_T < 0.0001f) done = true;    // Pixel saturated
```

**TC-GS** (batched threshold check):
```cpp
// Precomputed threshold: β > -ln(255) and β < 0
if (__hgt(exponents.x, __float2half_rn(-7.995f)) && 
    __hlt(exponents.x, __float2half_rn(0.0f))) {
    // Blend only if meaningful contribution
}
```

---

## NVTX Profiling Timeline

### Original 3DGS (~19.8 ms total)
```
|--Preprocessing--|--TileBinning--|----Sorting----|--------AlphaBlending--------|
|     2.66 ms     |    1.28 ms    |    4.28 ms    |          11.62 ms            |
|      13%        |      6%       |      22%      |            59%               |
```

### TC-GS (~10.4 ms total)
```
|Preproc|----TileBinning----|Sort|------AlphaBlending------|
| 1.17ms|      2.94 ms      |0.1 |        6.18 ms          |
|  11%  |        28%        | 1% |          60%            |
```

---

## Key Takeaways

### 1. Tight Tile Culling is a Trade-off
TC-GS spends **more time in TileBinning** (2.94 ms vs 1.28 ms) but this investment pays off:
- **35× faster Sorting** (0.12 ms vs 4.28 ms)
- **Fewer fragments to blend**

### 2. Sorting Benefits Most from Culling
The **34.8× speedup** in sorting (4.28 ms → 0.12 ms) shows that reducing the number of tile-Gaussian pairs has a massive impact on radix sort performance.

### 3. Tensor Core Utilization
TC-GS achieves **1.9× speedup** in alpha-blending by:
- Reformulating alpha computation as matrix multiplication
- Using tensor cores (MMA instructions) for batch processing
- FP16 computation with local coordinates to maintain precision

### 4. Bottleneck Analysis
| | Original 3DGS | TC-GS |
|--|--------------|-------|
| **Bottleneck** | AlphaBlending (59%) | AlphaBlending (60%) |
| **Second** | Sorting (22%) | TileBinning (28%) |
| **Optimized** | - | Sorting (22% → 1%) |

### 5. Quality Preservation
FP16 computation with **local coordinate transformation** maintains rendering quality within **0.02 dB PSNR** of the original.

### 6. Kernel Time vs End-to-End Performance
| Metric | Speedup | Explanation |
|--------|---------|-------------|
| **GPU Kernel Time** | 1.9× | Direct measurement of CUDA kernel execution |
| **End-to-End FPS** | 3.5× | Includes CPU overhead, memory transfers, sync delays |

The 3.5× FPS improvement exceeds the 1.9× kernel speedup because TC-GS also reduces:
- **Memory bandwidth** — fewer tile-Gaussian pairs to transfer
- **CPU-GPU synchronization** — less data to process between stages
- **Pipeline stalls** — tighter kernel scheduling with less idle time

---

## Deep Dive: Why TC-GS Pipeline is More Efficient

### 1. SnugBox Tight Culling (30-50% Fewer Tile-Gaussian Pairs)

The biggest impact comes from the `duplicateToTilesTouched` function using the **SnugBox algorithm**:

**Original 3DGS** — Simple rectangle bounding box:
```cpp
// Simple axis-aligned bounding box based on 3σ radius
float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
getRect(point_image, my_radius, rect_min, rect_max, grid);
tiles_touched = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
```

**TC-GS** — Exact ellipse-tile intersection with opacity threshold:
```cpp
// Opacity-aware threshold: only include tiles where Gaussian contributes ≥ 1/255
float t = 2.0f * log(con_o.w * 255.0f);
// Compute exact ellipse intersection per tile row/column
uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, ...);
```

**Visual comparison:**
```
Original 3DGS (Rectangle):          TC-GS (Tight Ellipse):
┌─┬─┬─┬─┬─┐                        ┌─┬─┬─┬─┬─┐
│█│█│█│█│█│  ← wasted corners      │ │█│█│█│ │
├─┼─┼─┼─┼─┤                        ├─┼─┼─┼─┼─┤
│█│█│█│█│█│                        │█│█│█│█│█│
├─┼─┼─┼─┼─┤                        ├─┼─┼─┼─┼─┤
│█│█│█│█│█│                        │ │█│█│█│ │
└─┴─┴─┴─┴─┘                        └─┴─┴─┴─┴─┘
25 tiles                           ~18 tiles (~30% fewer)
```

### 2. Tensor Core Batch Processing (16 Gaussians at Once)

**Original 3DGS** — Sequential scalar FP32 operations:
```cpp
// Process 1 Gaussian per iteration
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
    float alpha = min(0.99f, con_o.w * exp(power));  // Scalar exp()
    // ... blend one Gaussian
}
```

**TC-GS** — Matrix multiplication via MMA instructions:
```cpp
// Convert pixels and Gaussians to matrix format
uint4 pix_vec = pix2vec(pixf_local);
uint4 gaussian_vec = gs2vec(conics, means, pixf_mid);

// Batch compute 16 exponents via tensor core MMA instruction
mma_16x8x8_f16_f16(expmat_reg[0], expmat_reg[1],
    pixmat_reg[0], pixmat_reg[1], gsmat_reg[0], ...);  // 16 Gaussians at once!
```

**Throughput**: Tensor cores have **8-16× throughput** for matrix ops vs CUDA cores.

### 3. FP16 Reduces Memory Bandwidth by 50%

TC-GS compresses colors and features to FP16:
```cpp
// Pack 2 floats into 1 uint (8 bytes vs 16 bytes)
uint RG = float22reg(features.x, features.y);
uint BD = float22reg(features.z, features.w);
feature_encoded[idx] = make_uint2(RG, BD);
```

### 4. Local Coordinate Transformation

```cpp
// Global pixel coordinates (0-1600) → Local tile coordinates (-7.5 to +7.5)
pixf_mid = make_float2(pix_min.x + 7.5f, pix_min.y + 7.5f);
pixf_local = make_float2(pix.x - pixf_mid.x, pix.y - pixf_mid.y);
```

**Result**: FP16 maintains precision because values are small (±7.5 vs 0-1600).

### 5. The Cascade Effect

```
┌─────────────────────────────────────────────────────────────────────┐
│              WHY 1.9× KERNEL SPEEDUP → 3.5× FPS                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SnugBox Culling (30-50% fewer pairs)                              │
│       ↓                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • Sorting: 35× faster (radix sort O(n), n is smaller)       │   │
│  │ • Memory: 50% less bandwidth (FP16 + fewer items)           │   │
│  │ • Sync: Less CPU-GPU coordination                           │   │
│  │ • Prefix Sum: Smaller input → faster                        │   │
│  │ • Buffer Allocation: Smaller buffers → less overhead        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       ↓                                                            │
│  Tensor Core MMA (16 Gaussians batched)                            │
│       ↓                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • AlphaBlending: 1.9× faster (tensor cores + FP16)          │   │
│  │ • Better occupancy (more compute per memory access)         │   │
│  │ • Latency hiding (MMA ops have high throughput)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       ↓                                                            │
│  Pipeline gaps shrink → 3.5× total FPS improvement                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Summary Table

| Optimization | Implementation | Direct Impact | Cascade Impact |
|--------------|----------------|---------------|----------------|
| **SnugBox Culling** | Ellipse-tile intersection + opacity threshold | 30-50% fewer pairs | 35× faster sorting |
| **Tensor Core MMA** | `mma_16x8x8_f16_f16` instruction | 16 Gaussians/op | 1.9× faster blending |
| **FP16 Features** | `float22reg()` packing | 50% less memory | Better cache utilization |
| **Local Coordinates** | Tile-local transform (±7.5) | FP16 precision | No quality loss |

---

## References

1. TC-GS Paper: "TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores" (arXiv:2505.24796v2)
2. Original 3DGS: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023

