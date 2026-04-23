import torch
import torch.nn.functional as F
import cuda_ops._C as ops
import time

def benchmark(fn, *args, warmup=20, iters=100):
    # warmup — GPU needs a few runs to reach full speed
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    return elapsed

print("=" * 50)
print("BENCHMARKS vs PyTorch")
print("=" * 50)

# --- dot product ---
N = 10_000_000
a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")

ours    = benchmark(ops.dot_product, a, b)
pytorch = benchmark(torch.dot, a, b)
print(f"\nDot product (N={N:,})")
print(f"  ours:    {ours:.3f} ms")
print(f"  pytorch: {pytorch:.3f} ms")
print(f"  ratio:   {ours/pytorch:.2f}x")

# --- cross entropy ---
N, C = 10_000, 1_000
logits = torch.randn(N, C, device="cuda")
labels_int32 = torch.randint(0, C, (N,), device="cuda", dtype=torch.int32)
labels_int64 = labels_int32.long()

ours    = benchmark(ops.cross_entropy, logits, labels_int32)
pytorch = benchmark(F.cross_entropy, logits, labels_int64)
print(f"\nCross entropy (N={N:,}, C={C:,})")
print(f"  ours:    {ours:.3f} ms")
print(f"  pytorch: {pytorch:.3f} ms")
print(f"  ratio:   {ours/pytorch:.2f}x")

# --- matmul ---
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")

ours    = benchmark(ops.matmul, A, B)
pytorch = benchmark(torch.mm, A, B)
print(f"\nMatmul ({M}x{K} @ {K}x{N})")
print(f"  ours:    {ours:.3f} ms")
print(f"  pytorch: {pytorch:.3f} ms")
print(f"  ratio:   {ours/pytorch:.2f}x")

# --- prefix sum ---
N = 1_000_000
x = torch.randn(N, device="cuda")

ours    = benchmark(ops.prefix_sum, x)
pytorch = benchmark(torch.cumsum, x, 0)
print(f"\nPrefix sum (N={N:,})")
print(f"  ours:    {ours:.3f} ms")
print(f"  pytorch: {pytorch:.3f} ms")
print(f"  ratio:   {ours/pytorch:.2f}x")

print("\n" + "=" * 50)