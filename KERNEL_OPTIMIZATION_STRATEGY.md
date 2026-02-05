# Kernel Optimization Strategy: Key to High Emissions

## The Competition

You are competing to write the **fastest Triton kernels** for QuasarAttention. Your goal: maximize **tokens/sec** while passing **logit verification**.

## Understanding the New Scoring System

### How Validators Score You

```
weighted_score = tokens_per_sec × league_multiplier
```

**Top 4 miners share rewards:**
| Place | Reward Share |
|-------|--------------|
| 1st   | **60%**      |
| 2nd   | **25%**      |
| 3rd   | **10%**      |
| 4th   | **5%**       |

### League Multipliers (Context Length Bonuses)

| Target Sequence Length | League | Multiplier |
|-----------------------|--------|------------|
| < 200k                | 100k   | 0.5x       |
| 200k - 299k           | 200k   | 0.75x      |
| 300k - 399k           | 300k   | 1.0x       |
| 400k - 499k           | 400k   | 1.25x      |
| 500k - 599k           | 500k   | 1.5x       |
| 600k - 699k           | 600k   | 1.75x      |
| 700k - 799k           | 700k   | 2.0x       |
| 800k - 899k           | 800k   | 2.25x      |
| 900k - 999k           | 900k   | 2.5x       |
| >= 1M                 | 1M     | **3.0x**   |

**Example:** 50,000 tokens/sec at 1M sequence = 150,000 weighted score

### Ranking Criteria (In Order)

1. **Logit verification must PASS** (failed = excluded entirely)
2. **Weighted score** (descending)
3. **Submission timestamp** (ascending - **first wins ties**)
4. **Submission ID** (final tiebreaker)

## Critical Requirements

### 1. Logit Verification (Anti-Cheat Gate)

Validators compare your model's logits against a reference model at a random decode step:

| Metric | Threshold | Fail = |
|--------|-----------|--------|
| Cosine similarity | >= 0.99 | Score = 0 |
| Max absolute diff | <= 0.1 | Score = 0 |

**Reference model:** `Qwen/Qwen2.5-0.5B-Instruct`

**If you fail logit verification, you are excluded from rankings entirely.**

### 2. Performance Validation

The validator clones your repo, checks out your commit, and runs benchmarks:

- Must achieve **>= 90% of claimed performance** (10% tolerance)
- Tests run at: 512, 1024, 2048, and your `target_sequence_length`
- If actual < claimed × 0.9 → **Score = 0**

### 3. Import Validation (Hard Requirement)

Your code in `fla/ops/quasar/chunk.py` **MUST include**:

```python
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.quasar.forward_substitution import forward_substitution_kernel
from fla.utils import IS_AMD
from fla.utils import autocast_custom_bwd
from fla.utils import autocast_custom_fwd
from fla.utils import autotune_cache_kwargs
from fla.utils import check_shared_mem
from fla.utils import input_guard
```

**FORBIDDEN imports (will fail validation):**

```python
# DO NOT USE THESE - instant failure
from fla.ops.gla ...
from fla.ops.kda ...
import fla.ops.gla
import fla.ops.kda
```

## Target Files to Optimize

Located in `fla/ops/quasar/`:

| File | Priority | Description |
|------|----------|-------------|
| `chunk.py` | **1** | Core chunked attention - optimize first |
| `fused_recurrent.py` | **2** | Recurrent form for inference |
| `forward_substitution.py` | **3** | Forward substitution kernel |
| `chunk_intra_token_parallel.py` | **4** | Intra-token parallelization |
| `gate.py` | **5** | Gate mechanism |

## Optimization Strategy

### Phase 1: Baseline Establishment

```bash
# 1. Fork the repository
git clone https://github.com/troy12x/flash-linear-attention
cd flash-linear-attention

# 2. Run baseline benchmarks
python -c "
import torch
import time
from fla.layers.quasar import QuasarAttention

def benchmark(seq_len):
    model = QuasarAttention(
        hidden_size=512, head_dim=64, num_heads=8, mode='chunk'
    ).cuda().eval()
    x = torch.randn(1, seq_len, 512, device='cuda')

    # Warmup
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(10):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()

    return (seq_len * 10) / (time.time() - start)

for seq in [4096, 16384, 65536, 100000]:
    print(f'{seq}: {benchmark(seq):,.0f} tok/s')
"
```

### Phase 2: Quick Wins (10-50% Improvement)

#### Memory Access Optimization

```python
# Bad: Strided access
for i in range(N):
    out[i] = data[i * stride]

# Good: Coalesced access
@triton.jit
def kernel(data_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    data = tl.load(data_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data, mask=mask)
```

#### Tile Size Auto-Tuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def optimized_kernel(...):
    ...
```

#### Precision Optimization

- Use `bfloat16` for compute (validator uses `torch.autocast`)
- Use `float32` for accumulators to maintain numerical stability
- Avoid unnecessary type conversions

### Phase 3: Advanced Optimizations (2-3x Improvement)

#### Kernel Fusion

```python
# Before: Multiple kernel launches
q = linear_q(x)      # Kernel 1
k = linear_k(x)      # Kernel 2
v = linear_v(x)      # Kernel 3
attn = attention(q, k, v)  # Kernel 4

# After: Single fused kernel
@triton.jit
def fused_qkv_attention(...):
    # Load x once, compute q/k/v in registers
    # Compute attention, write output once
```

#### Chunked Processing for Long Sequences

```python
CHUNK_SIZE = 2048

for chunk_start in range(0, seq_len, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, seq_len)
    # Full attention within chunk
    # Linear attention approximation across chunks
```

### Phase 4: Hardware-Specific Tuning (10-30% Additional Gains)

**Goal:** Extract maximum performance from your specific GPU architecture.

#### 4.1 Know Your Hardware

| GPU | SMs | Shared Mem/SM | Registers/SM | HBM Bandwidth | Recommended BLOCK_M |
|-----|-----|---------------|--------------|---------------|---------------------|
| H100 SXM | 132 | 228 KB | 65536 | 3.35 TB/s | 128-256 |
| H100 PCIe | 114 | 228 KB | 65536 | 2.0 TB/s | 128-256 |
| A100 SXM | 108 | 164 KB | 65536 | 2.0 TB/s | 128-256 |
| A100 PCIe | 108 | 164 KB | 65536 | 1.6 TB/s | 128-256 |
| RTX 4090 | 128 | 100 KB | 65536 | 1.0 TB/s | 64-128 |
| RTX 3090 | 82 | 100 KB | 65536 | 936 GB/s | 64-128 |
| RTX 3080 | 68 | 100 KB | 65536 | 760 GB/s | 64-128 |

#### 4.2 Memory Hierarchy Optimization

```
Global Memory (HBM): 1-3 TB/s bandwidth, ~400 cycle latency
    ↓
L2 Cache: ~5 TB/s, ~100 cycle latency
    ↓
Shared Memory (SRAM): ~19 TB/s, ~30 cycle latency
    ↓
Registers: Instant access, limited quantity (255 per thread max)
```

**Strategy:** Keep frequently accessed data in registers/shared memory.

```python
@triton.jit
def optimized_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qm, stride_qk,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Load Q tile into registers (fast reuse)
    q_tile = tl.load(q_ptr + offsets_q)

    # Accumulate in registers, not global memory
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load K, V into shared memory for reuse
        k_tile = tl.load(k_ptr + offsets_k)
        v_tile = tl.load(v_ptr + offsets_v)

        # Compute in registers
        acc += tl.dot(q_tile, k_tile)

    # Single write to global memory
    tl.store(out_ptr + offsets_out, acc)
```

#### 4.3 Occupancy Optimization

Higher occupancy = better latency hiding. Target 50%+ occupancy.

```python
# Check kernel occupancy
import triton

@triton.jit
def my_kernel(...):
    ...

# Profile occupancy
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
kernel_info = triton.compile(my_kernel, signature="...", constants={...})
print(f"Registers per thread: {kernel_info.num_regs}")
print(f"Shared memory: {kernel_info.shared} bytes")

# Calculate theoretical occupancy
regs_per_thread = kernel_info.num_regs
threads_per_block = BLOCK_M * BLOCK_N // 32  # warps * 32
max_blocks_by_regs = 65536 // (regs_per_thread * threads_per_block)
max_blocks_by_shmem = 100 * 1024 // kernel_info.shared  # RTX 4090
occupancy = min(max_blocks_by_regs, max_blocks_by_shmem) / max_blocks_per_sm
```

**Tuning levers:**
- Reduce register usage → more concurrent warps
- Reduce shared memory → more concurrent blocks
- Increase BLOCK size → fewer blocks but more work per block

#### 4.4 Warp-Level Primitives

Use warp-level operations for faster reductions:

```python
@triton.jit
def fast_reduction_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(x_ptr + offsets, mask=offsets < N)

    # Warp-level reduction (much faster than global)
    result = tl.sum(x, axis=0)

    # Only first thread writes
    if tl.program_id(0) == 0:
        tl.store(out_ptr, result)
```

#### 4.5 Async Memory Operations (H100/A100)

```python
@triton.jit
def async_kernel(a_ptr, b_ptr, c_ptr, ...):
    # Prefetch next tile while computing current
    # (Triton handles this automatically with proper tiling)

    for i in range(num_tiles):
        # Load tile i+1 (async, overlaps with compute)
        next_tile = tl.load(a_ptr + next_offsets)

        # Compute on current tile
        result = tl.dot(current_a, current_b)

        # Swap tiles
        current_a = next_tile
```

#### 4.6 Profiling Tools

```bash
# NVIDIA Nsight Compute (detailed kernel analysis)
ncu --set full python benchmark.py

# NVIDIA Nsight Systems (timeline view)
nsys profile python benchmark.py

# PyTorch Profiler (integrated)
python -c "
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(5):
        model(x)
        prof.step()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
"

# Triton autotuning debug
export TRITON_PRINT_AUTOTUNING=1
python benchmark.py
```

#### 4.7 GPU-Specific Configurations

**For H100:**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
```

**For A100:**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
```

**For RTX 4090/3090:**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
```

#### 4.8 Long Sequence Memory Management

For 1M+ tokens, memory is critical:

```python
def memory_efficient_forward(x, chunk_size=65536):
    """Process ultra-long sequences in chunks to avoid OOM."""
    seq_len = x.shape[1]
    outputs = []

    # Running state for linear attention
    state = None

    for i in range(0, seq_len, chunk_size):
        chunk = x[:, i:i+chunk_size, :]

        # Process chunk with state carryover
        out, state = chunk_attention(chunk, state)
        outputs.append(out)

        # Free intermediate memory
        del chunk
        torch.cuda.empty_cache()

    return torch.cat(outputs, dim=1)
```

#### 4.9 Mixed Precision Strategy

```python
# Compute path: bfloat16 (faster, less memory)
# Accumulation: float32 (numerical stability)
# Output: bfloat16 (matches validator expectations)

@triton.jit
def mixed_precision_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    ...
):
    # Load as bfloat16
    q = tl.load(q_ptr + q_offsets).to(tl.float32)  # Upcast for compute
    k = tl.load(k_ptr + k_offsets).to(tl.float32)

    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(q, k)

    # Store as bfloat16
    tl.store(out_ptr + out_offsets, acc.to(tl.bfloat16))
```

#### 4.10 Benchmark Your Specific Hardware

```python
import torch
import subprocess

def get_gpu_info():
    """Get GPU specifications for tuning decisions."""
    props = torch.cuda.get_device_properties(0)
    return {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'sm_count': props.multi_processor_count,
        'global_memory_gb': props.total_memory / 1e9,
        'shared_memory_per_block': props.max_shared_memory_per_block,
        'registers_per_block': props.regs_per_block,
        'warp_size': props.warp_size,
        'max_threads_per_block': props.max_threads_per_block,
    }

gpu_info = get_gpu_info()
print(f"GPU: {gpu_info['name']}")
print(f"SMs: {gpu_info['sm_count']}")
print(f"Shared Memory/Block: {gpu_info['shared_memory_per_block'] / 1024:.0f} KB")
print(f"Registers/Block: {gpu_info['registers_per_block']}")

# Determine optimal config based on GPU
if 'H100' in gpu_info['name']:
    recommended = {'BLOCK_M': 256, 'BLOCK_N': 128, 'num_stages': 4}
elif 'A100' in gpu_info['name']:
    recommended = {'BLOCK_M': 128, 'BLOCK_N': 128, 'num_stages': 3}
else:  # Consumer GPUs
    recommended = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 2}

print(f"Recommended config: {recommended}")
```

## Submission Process

### Option 1: Direct Submission

```bash
export GITHUB_TOKEN="your_token"
export VALIDATOR_API_URL="https://quasar-subnet.onrender.com"

python neurons/miner.py \
  --wallet.name miner --wallet.hotkey default \
  --subtensor.network finney --netuid 439
```

### Option 2: Commit-Reveal (Recommended for IP Protection)

Prevents validators from copying your code before evaluation:

1. **Commit phase:** Submit `hash(salt + fork_url)`
2. **Wait:** ~100 blocks (~20 minutes)
3. **Reveal phase:** Submit actual fork URL + salt

## Benchmarking Script

```python
import torch
import time
import json
from fla.layers.quasar import QuasarAttention

def benchmark_all():
    results = {}

    for seq_len in [4096, 16384, 65536, 100000, 500000, 1000000]:
        try:
            model = QuasarAttention(
                hidden_size=512, head_dim=64, num_heads=8, mode='chunk'
            ).cuda().eval()

            x = torch.randn(1, seq_len, 512, device='cuda')

            # Warmup
            for _ in range(3):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(x)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()

            # Benchmark
            start = time.time()
            for _ in range(10):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            tokens_per_sec = (seq_len * 10) / elapsed
            vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            results[seq_len] = {
                'tokens_per_sec': tokens_per_sec,
                'vram_mb': vram_mb
            }
            print(f'{seq_len}: {tokens_per_sec:,.0f} tok/s | {vram_mb:.0f} MB')

        except torch.cuda.OutOfMemoryError:
            print(f'{seq_len}: OOM')
            break

    return results

if __name__ == '__main__':
    results = benchmark_all()
    print(json.dumps(results, indent=2))
```

## Common Pitfalls

### 1. Failing Logit Verification

Your optimizations must produce **numerically identical** outputs (within tolerance):

```python
def verify_correctness(optimized_fn, reference_fn, q, k, v, g):
    out_opt = optimized_fn(q, k, v, g)
    out_ref = reference_fn(q, k, v, g)

    # Must pass these checks
    assert torch.allclose(out_opt, out_ref, rtol=1e-3, atol=1e-3)
```

### 2. Claiming Inflated Performance

Validators run your code themselves. If actual < claimed × 0.9, you get **zero score**.

### 3. Forbidden Imports

Using `fla.ops.gla` or `fla.ops.kda` = instant validation failure.

### 4. Missing Required Imports

All required imports must be present in `chunk.py`.

### 5. Memory Blowup at Long Sequences

100k tokens × 2048 hidden × 4 bytes = 800 MB per tensor. Don't allocate unnecessary intermediates.

### 6. Forgetting CUDA Sync

```python
# WRONG
start = time.time()
kernel(...)
elapsed = time.time() - start  # Only measures launch time!

# RIGHT
start = time.time()
kernel(...)
torch.cuda.synchronize()  # Wait for GPU
elapsed = time.time() - start
```

## Pre-Submission Checklist

- [ ] All required imports present in `chunk.py`
- [ ] No forbidden imports anywhere
- [ ] Correctness verified against reference implementation
- [ ] Benchmarked at target sequence lengths
- [ ] Performance claims are achievable (within 10%)
- [ ] Code committed and pushed to fork
- [ ] CUDA synchronized during benchmarking

## Optimization Impact Summary

| Optimization | Expected Gain | Difficulty |
|--------------|---------------|------------|
| Use bfloat16 | 1.5-2x | Easy |
| Tune BLOCK sizes | 10-30% | Easy |
| Coalesced memory access | 20-50% | Medium |
| Kernel fusion | 30-100% | Medium |
| Custom Triton kernels | 2-5x | Hard |
| Hardware-specific tuning | 10-30% | Hard |

## The Winning Formula

```
High Emissions = (Fast Kernels × High League Multiplier) + First Submission
```

**Key strategies:**

1. **Maximize weighted score** - Target highest sequence length you can handle reliably
2. **Submit early** - First submission wins ties
3. **Pass verification** - Failed verification = excluded entirely
4. **Don't inflate claims** - Validators will verify

**The math is clear:** 50k tok/s at 1M context (3x multiplier) beats 100k tok/s at 300k context (1x multiplier).

## Resources

### Triton
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Flash Attention Implementation](https://github.com/Dao-AILab/flash-attention)

### Linear Attention
- [Linear Transformers Paper](https://arxiv.org/abs/2006.16236)
- [RWKV Architecture](https://arxiv.org/abs/2305.13048)
- [Mamba/State Space Models](https://arxiv.org/abs/2312.00752)

### GPU Programming
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
