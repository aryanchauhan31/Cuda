# CUDA Ops — Custom GPU Kernels as a PyTorch Extension

A collection of GPU kernels written from scratch in CUDA C++, packaged as a PyTorch C++ extension. The goal is not to beat cuBLAS — it's to deeply understand what happens inside the tools we use every day.

## Why This Exists

Most ML engineers use PyTorch's `torch.mm`, `torch.cumsum`, and `F.cross_entropy` without knowing what's inside. This project implements those operations from scratch in CUDA to understand the memory hierarchy, warp-level primitives, and optimization techniques that make GPUs fast.

Every kernel here was written by hand — no cuBLAS, no CUB, no cuDNN.

## What's Inside

| Kernel | Description | Technique |
|--------|-------------|-----------|
| **Dot Product** | Reduction-based inner product | Warp shuffle reduction, grid-stride loop, atomicAdd |
| **Matrix Multiply** | Tiled GEMM | Shared memory tiling (16×16), coalesced access |
| **Prefix Sum** | Inclusive scan | Kogge-Stone algorithm, recursive multi-block |
| **Cross Entropy** | Numerically stable CE loss | Log-sum-exp trick, block-level max/sum reduction, double accumulator |
| **2D Convolution** | Tiled conv with constant memory kernel | Shared memory halo loading, `__constant__` memory |
| **Reductions (1–4)** | Progressively optimized sum reductions | Naive → sequential addressing → warp shuffle → full warp+block |
| **Sparse MatMul** | CSR sparse × dense multiply | Row-parallel decomposition |

## Benchmarks vs PyTorch (T4 GPU)

All ops produce correct results verified against PyTorch reference implementations.

```
=== Correctness ===
dot product    ✓ matches torch.dot
matmul         ✓ matches torch.mm
prefix sum     ✓ matches torch.cumsum

=== Performance ===
op             ours       pytorch    ratio
dot product    0.296 ms   0.307 ms   0.97x  (match)
matmul         2.670 ms   0.470 ms   5.68x  (slower — no tensor cores, TILE=16)
prefix sum     0.378 ms   0.039 ms   9.74x  (slower — recursive malloc, no warp scan)
```

**Why the gaps exist (and what closing them would require):**

- **Dot product** is memory-bandwidth bound — both implementations hit the same bottleneck, so performance is equal.
- **Matmul** uses basic 16×16 shared memory tiling. PyTorch calls cuBLAS which uses 128×128 tiles, tensor cores, register-level blocking, and per-architecture autotuning. Closing the gap requires thread coarsening and double buffering.
- **Prefix sum** uses recursive Kogge-Stone with `cudaMalloc` at each recursion level. PyTorch uses CUB's decoupled lookback algorithm with zero allocations and warp-level scan primitives.

Understanding these gaps is the point — not hiding them.

## Installation

Requires a CUDA-capable GPU, PyTorch, and nvcc.

```bash
git clone https://github.com/aryanchauhan31/Cuda
cd Cuda
pip install -e .
```

## Usage

```python
import torch
import cuda_ops._C as ops

# dot product
a = torch.randn(10_000_000, device="cuda")
b = torch.randn(10_000_000, device="cuda")
result = ops.dot_product(a, b)

# matrix multiply
A = torch.randn(1024, 1024, device="cuda")
B = torch.randn(1024, 1024, device="cuda")
C = ops.matmul(A, B)

# prefix sum
x = torch.randn(1_000_000, device="cuda")
y = ops.prefix_sum(x)

# cross entropy loss
logits = torch.randn(100, 10, device="cuda")
labels = torch.randint(0, 10, (100,), device="cuda", dtype=torch.int32)
loss = ops.cross_entropy(logits, labels)
```

## Project Structure

```
Cuda/
  kernels/                         # CUDA kernel source files
    categorical_cross_entropy.cu   # cross entropy with log-sum-exp
    dot_product.cu                 # warp-shuffle dot product
    matmul.cu                      # tiled matrix multiply
    prefix_sum.cu                  # recursive kogge-stone scan
    reduction1-4.cu                # progressively optimized reductions
    sparse_matmul_kernel.cu        # CSR sparse matmul
  csrc/
    bindings.cpp                   # pybind11 bindings for PyTorch
  cuda_ops/
    __init__.py                    # Python package
    ops.py                         # Python API wrappers
  setup.py                         # build configuration
  tests/
    test_ops.py                    # correctness tests vs PyTorch
  benchmarks/
    bench.py                       # performance benchmarks
```

## Key Concepts Demonstrated

- **Memory hierarchy**: registers → shared memory → global memory, and when to use each
- **Coalesced access**: consecutive threads reading consecutive addresses for maximum bandwidth
- **Shared memory tiling**: loading tiles cooperatively to reduce global memory traffic
- **Bank conflict avoidance**: +1 padding on shared memory arrays
- **Warp-level primitives**: `__shfl_down_sync` for register-to-register communication without shared memory
- **Numerical stability**: max subtraction trick for softmax/cross-entropy to prevent overflow
- **Precision management**: double accumulators for large reductions to avoid float32 drift

## Roadmap

- [ ] Online softmax (2-pass instead of 3-pass)
- [ ] Simplified Flash Attention forward pass
- [ ] Fused conv + ReLU kernel
- [ ] Thread coarsening for matmul (4×4 per thread)
- [ ] Warp-level prefix scan to replace shared memory scan
- [ ] Nsight Compute profiling results

## Built With

- CUDA C++ (compute capability 7.5+)
- PyTorch C++ Extension API (`torch.utils.cpp_extension`)
- Tested on NVIDIA T4 (Google Colab)
