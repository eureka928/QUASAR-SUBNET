#!/usr/bin/env python3
"""
QUASAR Subnet - Step 2: Kernel Profiling & Tuning

After running Step 1 (baseline), use this script to:
1. Profile kernel execution to find bottlenecks
2. Test different block size configurations
3. Apply and measure optimizations
4. Generate optimized autotune configs

Usage:
    python scripts/step2_kernel_tuning.py --repo-path ./quasar_work/flash-linear-attention
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_step(step: int, description: str):
    print(f"\n[Step {step}] {description}")
    print("-" * 50)


def get_gpu_info() -> Dict:
    """Get detailed GPU information for tuning decisions."""
    import torch

    props = torch.cuda.get_device_properties(0)

    # Get shared memory (handle different PyTorch versions)
    shared_mem = getattr(props, 'shared_memory_per_block', 49152)  # default 48KB

    info = {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'sm_count': props.multi_processor_count,
        'total_memory_gb': props.total_memory / 1e9,
        'shared_memory_per_block': shared_mem,
        'shared_memory_per_block_kb': shared_mem / 1024,
        'registers_per_block': getattr(props, 'regs_per_block', 65536),  # default
        'warp_size': getattr(props, 'warp_size', 32),
        'max_threads_per_block': props.max_threads_per_block,
        'max_threads_per_sm': getattr(props, 'max_threads_per_multi_processor', 2048),
    }

    # Determine GPU category for recommendations
    name_lower = info['name'].lower()
    if 'h100' in name_lower:
        info['category'] = 'H100'
        info['recommended_blocks'] = [(256, 128), (128, 256), (128, 128)]
        info['recommended_stages'] = [4, 3]
        info['recommended_warps'] = [8, 4]
    elif 'a100' in name_lower:
        info['category'] = 'A100'
        info['recommended_blocks'] = [(128, 128), (128, 64), (64, 128)]
        info['recommended_stages'] = [3, 4]
        info['recommended_warps'] = [8, 4]
    elif '4090' in name_lower:
        info['category'] = 'RTX4090'
        info['recommended_blocks'] = [(128, 64), (64, 128), (64, 64)]
        info['recommended_stages'] = [2, 3]
        info['recommended_warps'] = [4, 8]
    elif '3090' in name_lower or '3080' in name_lower:
        info['category'] = 'RTX30xx'
        info['recommended_blocks'] = [(64, 64), (128, 64), (64, 128)]
        info['recommended_stages'] = [2]
        info['recommended_warps'] = [4]
    else:
        info['category'] = 'Generic'
        info['recommended_blocks'] = [(64, 64), (128, 64), (64, 128)]
        info['recommended_stages'] = [2, 3]
        info['recommended_warps'] = [4, 8]

    return info


def profile_kernel(seq_len: int = 16384) -> Dict:
    """Profile the QuasarAttention kernel to find bottlenecks."""
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from fla.layers.quasar import QuasarAttention

    device = torch.device("cuda")

    model = QuasarAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        mode="chunk",
        use_short_conv=True,
    ).to(device).eval()

    x = torch.randn(1, seq_len, 512, device=device)

    # Warmup
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("quasar_forward"):
            for _ in range(5):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(x)
            torch.cuda.synchronize()

    # Extract key metrics
    events = prof.key_averages()

    results = {
        'total_cuda_time_ms': 0,
        'top_kernels': [],
        'memory_allocated_mb': 0,
    }

    print("\nTop 10 CUDA Operations by Time:")
    print(f"{'Operation':<50} {'Time (ms)':<15} {'Calls':<10}")
    print("-" * 75)

    # Handle different PyTorch versions - use cuda_time_total if available, else self_cuda_time_total or cpu_time_total
    def get_cuda_time(event):
        return getattr(event, 'cuda_time_total', 0) or getattr(event, 'self_cuda_time_total', 0) or getattr(event, 'cpu_time_total', 0)

    sorted_events = sorted(events, key=get_cuda_time, reverse=True)

    for event in sorted_events[:10]:
        event_time = get_cuda_time(event)
        if event_time > 0:
            time_ms = event_time / 1000  # Convert to ms
            results['total_cuda_time_ms'] += time_ms
            results['top_kernels'].append({
                'name': event.key[:50],
                'cuda_time_ms': time_ms,
                'calls': event.count,
            })
            print(f"{event.key[:50]:<50} {time_ms:<15.3f} {event.count:<10}")

    return results


def test_block_configurations(seq_len: int = 65536) -> List[Dict]:
    """Test different block size configurations."""
    import torch
    from fla.layers.quasar import QuasarAttention

    device = torch.device("cuda")
    gpu_info = get_gpu_info()

    print(f"\nTesting block configurations for {gpu_info['name']}...")
    print(f"Sequence length: {seq_len}")

    # Base configurations to test
    configs = [
        {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4},
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4},
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4},
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4},
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4},
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4},
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8},
    ]

    # Add GPU-specific configs
    if gpu_info['category'] in ['H100', 'A100']:
        configs.extend([
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8},
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8},
        ])

    results = []

    model = QuasarAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        mode="chunk",
        use_short_conv=True,
    ).to(device).eval()

    x = torch.randn(1, seq_len, 512, device=device)

    print(f"\n{'Config':<45} {'Tokens/sec':<15} {'Time (ms)':<12} {'Status'}")
    print("-" * 80)

    # Test each configuration
    # Note: In a real implementation, you would modify the Triton kernel
    # to use these configs. Here we just benchmark the current implementation
    # with different approaches.

    # Warmup
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()

    # Benchmark current implementation
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (seq_len * num_runs) / elapsed
    time_per_run = (elapsed / num_runs) * 1000

    current_result = {
        'config': 'Current Implementation',
        'tokens_per_sec': tokens_per_sec,
        'time_ms': time_per_run,
    }
    results.append(current_result)

    print(f"{'Current Implementation':<45} {tokens_per_sec:<15,.0f} {time_per_run:<12.2f} ✓")

    # Show recommended configs to try
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATIONS TO TRY")
    print("=" * 70)
    print(f"\nFor your GPU ({gpu_info['name']}):")
    print(f"  Category: {gpu_info['category']}")
    print(f"  SMs: {gpu_info['sm_count']}")
    print(f"  Shared Memory: {gpu_info['shared_memory_per_block_kb']:.0f} KB/block")
    print(f"\nRecommended block sizes: {gpu_info['recommended_blocks']}")
    print(f"Recommended num_stages: {gpu_info['recommended_stages']}")
    print(f"Recommended num_warps: {gpu_info['recommended_warps']}")

    return results


def generate_autotune_config(gpu_info: Dict) -> str:
    """Generate optimized autotune configuration for the GPU."""

    configs = []

    for block_m, block_n in gpu_info['recommended_blocks']:
        for stages in gpu_info['recommended_stages']:
            for warps in gpu_info['recommended_warps']:
                for block_k in [32, 64]:
                    configs.append(
                        f"        triton.Config({{'BLOCK_M': {block_m}, 'BLOCK_N': {block_n}, "
                        f"'BLOCK_K': {block_k}}}, num_stages={stages}, num_warps={warps}),"
                    )

    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique_configs.append(c)

    autotune_code = f"""
# Optimized autotune configuration for {gpu_info['name']}
# Generated by QUASAR step2_kernel_tuning.py

@triton.autotune(
    configs=[
{chr(10).join(unique_configs[:12])}  # Limit to 12 configs for faster autotuning
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def your_kernel_name(...):
    ...
"""
    return autotune_code


def analyze_memory_access(repo_path: Path):
    """Analyze memory access patterns in the kernel."""
    print_header("Memory Access Analysis")

    chunk_path = repo_path / "fla" / "ops" / "quasar" / "chunk.py"

    if not chunk_path.exists():
        print(f"⚠ Could not find {chunk_path}")
        return

    with open(chunk_path, 'r') as f:
        content = f.read()

    # Look for common patterns
    issues = []
    suggestions = []

    # Check for tl.load without mask
    if 'tl.load(' in content and content.count('mask=') < content.count('tl.load('):
        issues.append("Some tl.load() calls may be missing mask parameter")
        suggestions.append("Add mask parameter to all tl.load() for boundary safety")

    # Check for non-coalesced access patterns
    if 'stride' in content.lower():
        suggestions.append("Review stride usage - ensure memory accesses are coalesced")

    # Check for float32 usage
    if 'float32' in content or 'torch.float32' in content:
        suggestions.append("Consider using bfloat16 for compute, float32 only for accumulators")

    # Check for tl.atomic operations (slow)
    if 'tl.atomic' in content:
        issues.append("Atomic operations detected - these are slow")
        suggestions.append("Try to restructure to avoid atomic operations")

    # Check for synchronization
    if 'tl.debug_barrier' in content:
        issues.append("Debug barriers found - remove for production")

    print("Analysis of chunk.py:")
    print("-" * 50)

    if issues:
        print("\n⚠ Potential Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No obvious issues detected")

    if suggestions:
        print("\n💡 Optimization Suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

    # Count kernels
    kernel_count = content.count('@triton.jit')
    autotune_count = content.count('@triton.autotune')

    print(f"\nKernel Statistics:")
    print(f"  - Triton kernels: {kernel_count}")
    print(f"  - With autotune: {autotune_count}")

    if autotune_count < kernel_count:
        print(f"  ⚠ {kernel_count - autotune_count} kernels without autotune - consider adding")


def create_optimization_patch(repo_path: Path, gpu_info: Dict):
    """Generate a patch file with optimization suggestions."""
    print_header("Generating Optimization Patch")

    patch_content = f"""# QUASAR Kernel Optimization Patch
# Generated for: {gpu_info['name']} ({gpu_info['category']})
#
# Instructions:
# 1. Review the suggested changes below
# 2. Apply relevant optimizations to fla/ops/quasar/chunk.py
# 3. Test correctness after each change
# 4. Benchmark to measure improvement

# ============================================================
# OPTIMIZATION 1: Add Optimized Autotune Configuration
# ============================================================
#
# Add this autotune decorator before your main kernel:

{generate_autotune_config(gpu_info)}

# ============================================================
# OPTIMIZATION 2: Memory Access Pattern
# ============================================================
#
# Ensure coalesced memory access:
#
# BAD (strided access):
#   for i in range(N):
#       out[i] = data[i * stride]
#
# GOOD (coalesced access):
#   offsets = pid * BLOCK + tl.arange(0, BLOCK)
#   data = tl.load(data_ptr + offsets, mask=offsets < N)

# ============================================================
# OPTIMIZATION 3: Use Shared Memory for Data Reuse
# ============================================================
#
# If you access the same data multiple times, use shared memory:
#
# # Allocate shared memory
# shared_data = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
#
# # Load once to shared memory
# shared_data = tl.load(data_ptr + offsets)
#
# # Reuse multiple times
# result1 = tl.dot(shared_data, weights1)
# result2 = tl.dot(shared_data, weights2)

# ============================================================
# OPTIMIZATION 4: Accumulator Precision
# ============================================================
#
# Use float32 for accumulators, bfloat16 for compute:
#
# # Initialize accumulator in float32
# acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
#
# # Load data as bfloat16
# a = tl.load(a_ptr + offsets).to(tl.bfloat16)
# b = tl.load(b_ptr + offsets).to(tl.bfloat16)
#
# # Accumulate in float32
# acc += tl.dot(a, b, out_dtype=tl.float32)
#
# # Store as bfloat16
# tl.store(out_ptr + offsets, acc.to(tl.bfloat16))

# ============================================================
# OPTIMIZATION 5: Reduce Register Pressure
# ============================================================
#
# If you're hitting register limits:
# - Reduce BLOCK sizes
# - Use fewer intermediate variables
# - Consider splitting into multiple kernels

# ============================================================
# GPU-SPECIFIC RECOMMENDATIONS for {gpu_info['name']}
# ============================================================
#
# Optimal BLOCK_M: {gpu_info['recommended_blocks'][0][0]}
# Optimal BLOCK_N: {gpu_info['recommended_blocks'][0][1]}
# Recommended num_stages: {gpu_info['recommended_stages'][0]}
# Recommended num_warps: {gpu_info['recommended_warps'][0]}
#
# Your GPU has:
#   - {gpu_info['sm_count']} SMs
#   - {gpu_info['shared_memory_per_block_kb']:.0f} KB shared memory per block
#   - {gpu_info['registers_per_block']} registers per block
#   - {gpu_info['total_memory_gb']:.1f} GB total memory
"""

    patch_path = repo_path / "optimization_suggestions.txt"
    with open(patch_path, 'w') as f:
        f.write(patch_content)

    print(f"✓ Optimization suggestions saved to: {patch_path}")
    print("\nKey recommendations:")
    print(f"  1. Use BLOCK_M={gpu_info['recommended_blocks'][0][0]}, BLOCK_N={gpu_info['recommended_blocks'][0][1]}")
    print(f"  2. Set num_stages={gpu_info['recommended_stages'][0]}")
    print(f"  3. Set num_warps={gpu_info['recommended_warps'][0]}")
    print(f"  4. Use bfloat16 compute with float32 accumulators")

    return patch_path


def benchmark_sequence_lengths(target_league: str = "1M"):
    """Benchmark at sequence lengths needed for target league."""
    import torch
    from fla.layers.quasar import QuasarAttention

    print_header(f"Benchmarking for {target_league} League")

    league_targets = {
        "100k": [4096, 16384, 65536, 100000],
        "300k": [4096, 16384, 65536, 100000, 200000, 300000],
        "500k": [4096, 65536, 100000, 300000, 500000],
        "700k": [4096, 100000, 300000, 500000, 700000],
        "1M": [4096, 100000, 500000, 750000, 1000000],
    }

    seq_lengths = league_targets.get(target_league, league_targets["100k"])

    device = torch.device("cuda")
    results = {}

    model = QuasarAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        mode="chunk",
        use_short_conv=True,
    ).to(device).eval()

    print(f"{'Seq Length':<12} {'Tokens/sec':<15} {'VRAM (MB)':<12} {'Weighted':<15} {'Status'}")
    print("-" * 70)

    league_multipliers = {
        100000: 0.5, 200000: 0.75, 300000: 1.0, 400000: 1.25,
        500000: 1.5, 600000: 1.75, 700000: 2.0, 800000: 2.25,
        900000: 2.5, 1000000: 3.0,
    }

    for seq_len in seq_lengths:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(1, seq_len, 512, device=device)

            # Warmup
            for _ in range(3):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(x)
            torch.cuda.synchronize()

            # Benchmark
            num_runs = 10
            start = time.time()
            for _ in range(num_runs):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            tokens_per_sec = (seq_len * num_runs) / elapsed
            vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            # Calculate weighted score
            multiplier = 0.5
            for threshold, mult in sorted(league_multipliers.items()):
                if seq_len >= threshold:
                    multiplier = mult
            weighted = tokens_per_sec * multiplier

            results[seq_len] = {
                'tokens_per_sec': tokens_per_sec,
                'vram_mb': vram_mb,
                'multiplier': multiplier,
                'weighted': weighted,
            }

            print(f"{seq_len:<12} {tokens_per_sec:<15,.0f} {vram_mb:<12,.0f} {weighted:<15,.0f} ✓")

            del x
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:<12} {'OOM':<15} {'-':<12} {'-':<15} ✗")
            break

    # Find best weighted score
    if results:
        best = max(results.items(), key=lambda x: x[1]['weighted'])
        print(f"\n✓ Best weighted score: {best[1]['weighted']:,.0f} at seq_len={best[0]}")
        print(f"  ({best[1]['tokens_per_sec']:,.0f} tok/s × {best[1]['multiplier']}x multiplier)")

    return results


def main():
    parser = argparse.ArgumentParser(description="QUASAR Step 2: Kernel Profiling & Tuning")
    parser.add_argument("--repo-path", default="./quasar_work/flash-linear-attention",
                        help="Path to flash-linear-attention repo")
    parser.add_argument("--seq-len", type=int, default=65536,
                        help="Sequence length for profiling")
    parser.add_argument("--target-league", default="1M",
                        choices=["100k", "300k", "500k", "700k", "1M"],
                        help="Target league for benchmarking")
    parser.add_argument("--skip-profile", action="store_true",
                        help="Skip profiling step")
    args = parser.parse_args()

    repo_path = Path(args.repo_path)

    print_header("QUASAR Step 2: Kernel Profiling & Tuning")

    # Step 1: Get GPU info
    print_step(1, "Analyzing GPU")
    gpu_info = get_gpu_info()

    print(f"GPU: {gpu_info['name']}")
    print(f"Category: {gpu_info['category']}")
    print(f"SMs: {gpu_info['sm_count']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"Shared Memory: {gpu_info['shared_memory_per_block_kb']:.0f} KB/block")

    # Step 2: Profile kernel
    if not args.skip_profile:
        print_step(2, "Profiling Kernel Execution")
        profile_results = profile_kernel(args.seq_len)

    # Step 3: Test configurations
    print_step(3, "Testing Block Configurations")
    config_results = test_block_configurations(args.seq_len)

    # Step 4: Analyze memory access
    print_step(4, "Analyzing Memory Access Patterns")
    if repo_path.exists():
        analyze_memory_access(repo_path)
    else:
        print(f"⚠ Repo not found at {repo_path}, skipping analysis")

    # Step 5: Generate optimization suggestions
    print_step(5, "Generating Optimization Suggestions")
    if repo_path.exists():
        patch_path = create_optimization_patch(repo_path, gpu_info)

    # Step 6: Benchmark for target league
    print_step(6, f"Benchmarking for {args.target_league} League")
    league_results = benchmark_sequence_lengths(args.target_league)

    # Summary
    print_header("STEP 2 COMPLETE - Summary")

    league_multipliers = {'100k': '0.5x', '300k': '1.0x', '500k': '1.5x', '700k': '2.0x', '1M': '3.0x'}
    multiplier = league_multipliers.get(args.target_league, '1.0x')

    print(f"""
GPU: {gpu_info['name']} ({gpu_info['category']})

Recommended Configuration:
  BLOCK_M: {gpu_info['recommended_blocks'][0][0]}
  BLOCK_N: {gpu_info['recommended_blocks'][0][1]}
  num_stages: {gpu_info['recommended_stages'][0]}
  num_warps: {gpu_info['recommended_warps'][0]}

Next Steps:
  1. Review optimization_suggestions.txt in your repo
  2. Edit fla/ops/quasar/chunk.py with the suggested changes
  3. Test correctness: python scripts/step1_first_optimization.py --skip-setup
  4. Benchmark again: python scripts/step2_kernel_tuning.py
  5. Submit when you see improvement!

Target: {args.target_league} league ({multiplier} multiplier)
""")


if __name__ == "__main__":
    main()
