#!/usr/bin/env python3
"""
QUASAR Subnet - Step 1: First Kernel Optimization

This script guides you through your first kernel optimization submission.
Run this script to:
1. Check your environment
2. Clone/setup the flash-linear-attention repo
3. Run baseline benchmarks
4. Apply a simple optimization (block size tuning)
5. Verify correctness
6. Prepare for submission

Usage:
    python scripts/step1_first_optimization.py --github-token YOUR_TOKEN
"""

import os
import sys
import subprocess
import time
import argparse
import json
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_step(step: int, description: str):
    """Print a step indicator."""
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


def check_environment():
    """Check that all required dependencies are available."""
    print_header("Environment Check")

    issues = []

    # Check Python version
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}", end=" ")
    if py_version >= (3, 9):
        print("✓")
    else:
        print("✗ (need 3.9+)")
        issues.append("Python 3.9+ required")

    # Check CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}", end=" ")
        if torch.cuda.is_available():
            print("✓")
            print(f"CUDA: {torch.version.cuda} ✓")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

            # Check GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {total_mem:.1f} GB", end=" ")
            if total_mem >= 16:
                print("✓ (recommended)")
            elif total_mem >= 8:
                print("⚠ (minimum, may limit sequence length)")
            else:
                print("✗ (may be insufficient)")
                issues.append(f"GPU has only {total_mem:.1f} GB VRAM")
        else:
            print("✗ (CUDA not available)")
            issues.append("CUDA not available")
    except ImportError:
        print("PyTorch: ✗ (not installed)")
        issues.append("PyTorch not installed")

    # Check Triton
    try:
        import triton
        print(f"Triton: {triton.__version__} ✓")
    except ImportError:
        print("Triton: ✗ (not installed)")
        issues.append("Triton not installed - run: pip install triton")

    # Check git
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        print(f"Git: {result.stdout.strip().split()[-1]} ✓")
    except FileNotFoundError:
        print("Git: ✗ (not installed)")
        issues.append("Git not installed")

    if issues:
        print(f"\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\n✓ Environment OK")
    return True


def setup_fork(github_token: str, github_username: str, work_dir: str):
    """Clone or update the flash-linear-attention fork."""
    print_header("Repository Setup")

    repo_url = "https://github.com/troy12x/flash-linear-attention"
    repo_path = Path(work_dir) / "flash-linear-attention"

    if repo_path.exists():
        print(f"Repository already exists at {repo_path}")
        print("Pulling latest changes...")
        subprocess.run(["git", "pull"], cwd=repo_path, check=True)
    else:
        print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)

    # Install the package in development mode
    print("\nInstalling flash-linear-attention in development mode...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_path)], check=True)

    print(f"\n✓ Repository ready at: {repo_path}")
    return repo_path


def run_baseline_benchmark(repo_path: Path):
    """Run baseline benchmarks to establish current performance."""
    print_header("Baseline Benchmark")

    import torch
    from fla.layers.quasar import QuasarAttention

    device = torch.device("cuda")
    results = {}

    # Model configuration (matches validator)
    hidden_size = 512
    head_dim = 64
    num_heads = 8

    model = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device).eval()

    # Test sequence lengths
    seq_lengths = [4096, 16384, 65536, 100000]

    # Check available memory and adjust if needed
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem < 16:
        seq_lengths = [4096, 16384, 32768]
        print(f"⚠ Limited GPU memory ({total_mem:.1f} GB), testing shorter sequences")

    print(f"Testing with hidden_size={hidden_size}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"\n{'Seq Length':<12} {'Tokens/sec':<15} {'VRAM (MB)':<12} {'Status'}")
    print("-" * 50)

    for seq_len in seq_lengths:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(1, seq_len, hidden_size, device=device)

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

            results[seq_len] = {
                'tokens_per_sec': tokens_per_sec,
                'vram_mb': vram_mb
            }

            print(f"{seq_len:<12} {tokens_per_sec:<15,.0f} {vram_mb:<12,.0f} ✓")

            del x
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:<12} {'OOM':<15} {'-':<12} ✗")
            break

    # Save baseline results
    baseline_file = repo_path / "baseline_results.json"
    with open(baseline_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Baseline saved to: {baseline_file}")
    return results


def find_chunk_py(repo_path: Path) -> Path:
    """Find the chunk.py file in the repository."""
    chunk_path = repo_path / "fla" / "ops" / "quasar" / "chunk.py"
    if chunk_path.exists():
        return chunk_path

    # Try to find it
    for path in repo_path.rglob("chunk.py"):
        if "quasar" in str(path):
            return path

    raise FileNotFoundError(f"Could not find chunk.py in {repo_path}")


def apply_simple_optimization(repo_path: Path):
    """Apply a simple optimization: enable Triton autotuning debug output."""
    print_header("First Optimization: Triton Autotuning Analysis")

    chunk_path = find_chunk_py(repo_path)
    print(f"Target file: {chunk_path}")

    # Read the current file
    with open(chunk_path, 'r') as f:
        content = f.read()

    # Check if autotuning is already configured
    if "@triton.autotune" in content:
        print("\n✓ Triton autotuning already present in chunk.py")
        print("\nTo optimize further, you can:")
        print("  1. Add more config options to the autotune decorator")
        print("  2. Adjust BLOCK_M, BLOCK_N, BLOCK_K values")
        print("  3. Change num_stages and num_warps")

        # Show current autotune configs
        print("\nCurrent autotune configurations found:")
        lines = content.split('\n')
        in_autotune = False
        for i, line in enumerate(lines):
            if '@triton.autotune' in line:
                in_autotune = True
            if in_autotune:
                print(f"  {line}")
                if 'def ' in line and '(' in line:
                    in_autotune = False
                    print()
    else:
        print("\n⚠ No @triton.autotune found - this is a basic implementation")
        print("\nYou can add autotuning by modifying the Triton kernels.")

    print("\n" + "=" * 60)
    print("OPTIMIZATION SUGGESTIONS")
    print("=" * 60)
    print("""
1. EASY WINS (10-30% improvement):
   - Ensure bfloat16 is used (validator uses torch.autocast)
   - Add more autotune configurations for your GPU

2. MEDIUM EFFORT (30-100% improvement):
   - Optimize memory access patterns (coalescing)
   - Tune BLOCK sizes for your specific GPU:
     * H100/A100: BLOCK_M=128-256, BLOCK_N=128-256
     * RTX 4090/3090: BLOCK_M=64-128, BLOCK_N=64-128
   - Increase num_stages for better pipelining

3. HIGH EFFORT (2-5x improvement):
   - Fuse multiple kernels into one
   - Implement custom memory management
   - Use warp-level primitives

Example autotune config to add:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
```
""")

    return chunk_path


def verify_correctness(repo_path: Path):
    """Verify that the optimized kernel produces correct results."""
    print_header("Correctness Verification")

    import torch
    from fla.layers.quasar import QuasarAttention

    device = torch.device("cuda")

    # Create model
    model = QuasarAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        mode="chunk",
        use_short_conv=True,
    ).to(device).eval()

    # Test with different inputs
    test_cases = [
        (1, 1024, 512),   # Small
        (1, 4096, 512),   # Medium
        (2, 2048, 512),   # Batch > 1
    ]

    print("Running correctness tests...")

    all_passed = True
    for batch, seq_len, hidden in test_cases:
        torch.manual_seed(42)  # Reproducible
        x = torch.randn(batch, seq_len, hidden, device=device)

        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output, _, _ = model(x)

            # Basic checks
            assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"

            print(f"  [{batch}, {seq_len}, {hidden}] ✓")

        except Exception as e:
            print(f"  [{batch}, {seq_len}, {hidden}] ✗ - {e}")
            all_passed = False

    if all_passed:
        print("\n✓ All correctness tests passed")
    else:
        print("\n✗ Some tests failed - DO NOT SUBMIT until fixed")

    return all_passed


def run_optimized_benchmark(repo_path: Path, baseline_results: dict):
    """Run benchmarks after optimization and compare to baseline."""
    print_header("Optimized Benchmark")

    import torch
    # Reimport to get any changes
    import importlib
    import fla.layers.quasar
    importlib.reload(fla.layers.quasar)
    from fla.layers.quasar import QuasarAttention

    device = torch.device("cuda")
    results = {}

    model = QuasarAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        mode="chunk",
        use_short_conv=True,
    ).to(device).eval()

    seq_lengths = list(baseline_results.keys())

    print(f"\n{'Seq Length':<12} {'Baseline':<15} {'Optimized':<15} {'Improvement'}")
    print("-" * 60)

    for seq_len in seq_lengths:
        seq_len = int(seq_len)  # JSON keys are strings
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

            results[seq_len] = {
                'tokens_per_sec': tokens_per_sec,
                'vram_mb': vram_mb
            }

            # Handle both int and string keys (int from memory, string from JSON)
            baseline_key = seq_len if seq_len in baseline_results else str(seq_len)
            baseline_tps = baseline_results[baseline_key]['tokens_per_sec']
            improvement = ((tokens_per_sec - baseline_tps) / baseline_tps) * 100

            status = "✓" if improvement > 0 else "→"
            print(f"{seq_len:<12} {baseline_tps:<15,.0f} {tokens_per_sec:<15,.0f} {improvement:+.1f}% {status}")

            del x
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:<12} {'OOM':<15} {'OOM':<15} -")
            break

    # Save optimized results
    optimized_file = repo_path / "optimized_results.json"
    with open(optimized_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {optimized_file}")
    return results


def prepare_submission(repo_path: Path, results: dict, github_username: str):
    """Prepare the submission payload."""
    print_header("Submission Preparation")

    # Get git info
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    commit_hash = result.stdout.strip()

    # Find best result (highest weighted score)
    league_multipliers = {
        100000: 0.5,   # 100k league
        200000: 0.75,
        300000: 1.0,
        400000: 1.25,
        500000: 1.5,
        600000: 1.75,
        700000: 2.0,
        800000: 2.25,
        900000: 2.5,
        1000000: 3.0,
    }

    best_seq_len = None
    best_weighted = 0
    best_tps = 0

    for seq_len_str, data in results.items():
        seq_len = int(seq_len_str)
        tps = data['tokens_per_sec']

        # Find applicable multiplier
        multiplier = 0.5  # Default to lowest
        for threshold, mult in sorted(league_multipliers.items()):
            if seq_len >= threshold:
                multiplier = mult

        weighted = tps * multiplier
        if weighted > best_weighted:
            best_weighted = weighted
            best_seq_len = seq_len
            best_tps = tps

    print(f"Best result: {best_tps:,.0f} tok/s at {best_seq_len} sequence length")
    print(f"Weighted score: {best_weighted:,.0f}")

    # Prepare benchmarks dict
    benchmarks = {}
    for seq_len_str, data in results.items():
        benchmarks[seq_len_str] = {
            'tokens_per_sec': data['tokens_per_sec'],
            'vram_mb': data['vram_mb']
        }

    # Handle both int and string keys
    best_key = best_seq_len if best_seq_len in results else str(best_seq_len)

    submission = {
        "fork_url": f"https://github.com/{github_username}/flash-linear-attention",
        "commit_hash": commit_hash,
        "target_sequence_length": best_seq_len,
        "tokens_per_sec": best_tps,
        "vram_mb": results[best_key]['vram_mb'],
        "benchmarks": benchmarks
    }

    print("\n" + "=" * 60)
    print("SUBMISSION PAYLOAD")
    print("=" * 60)
    print(json.dumps(submission, indent=2))

    # Save submission
    submission_file = repo_path / "submission_payload.json"
    with open(submission_file, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n✓ Submission payload saved to: {submission_file}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. PUSH YOUR CHANGES:
   cd {repo_path}
   git add -A
   git commit -m "Kernel optimization"
   git push origin main

2. FORK THE REPO (if not already done):
   - Go to https://github.com/troy12x/flash-linear-attention
   - Click "Fork" to create your own copy
   - Update the fork_url in submission_payload.json

3. SUBMIT TO VALIDATOR:
   python neurons/miner.py \\
     --wallet.name miner \\
     --wallet.hotkey default \\
     --subtensor.network finney \\
     --netuid 439

4. VERIFY YOUR SUBMISSION:
   curl https://quasar-subnet.onrender.com/get_submission_stats

IMPORTANT REMINDERS:
- Your claimed performance must be within 10% of actual
- Logit verification must pass (cosine sim >= 0.99)
- First submission wins ties - submit early!
- Top 4 miners share rewards: 60% / 25% / 10% / 5%
""")

    return submission


def main():
    parser = argparse.ArgumentParser(description="QUASAR Step 1: First Optimization")
    parser.add_argument("--github-token", help="GitHub personal access token")
    parser.add_argument("--github-username", default="", help="Your GitHub username")
    parser.add_argument("--work-dir", default="./quasar_work", help="Working directory")
    parser.add_argument("--skip-setup", action="store_true", help="Skip repo setup")
    args = parser.parse_args()

    print_header("QUASAR Subnet - First Optimization Guide")

    # Step 1: Environment check
    print_step(1, "Checking Environment")
    if not check_environment():
        print("\n⚠ Please fix the environment issues before continuing.")
        sys.exit(1)

    # Step 2: Setup repository
    if not args.skip_setup:
        print_step(2, "Setting Up Repository")
        github_token = args.github_token or os.environ.get("GITHUB_TOKEN", "")
        github_username = args.github_username or os.environ.get("GITHUB_USERNAME", "your_username")

        if not github_token:
            print("⚠ No GitHub token provided. Some operations may be limited.")
            print("  Set GITHUB_TOKEN environment variable or use --github-token")

        repo_path = setup_fork(github_token, github_username, args.work_dir)
    else:
        repo_path = Path(args.work_dir) / "flash-linear-attention"
        github_username = args.github_username or "your_username"

    # Step 3: Run baseline benchmark
    print_step(3, "Running Baseline Benchmark")
    baseline_results = run_baseline_benchmark(repo_path)

    # Step 4: Apply optimization
    print_step(4, "Analyzing Optimization Opportunities")
    chunk_path = apply_simple_optimization(repo_path)

    # Step 5: Verify correctness
    print_step(5, "Verifying Correctness")
    if not verify_correctness(repo_path):
        print("\n⚠ Correctness verification failed!")
        print("  Fix any issues before making optimizations.")

    # Step 6: Run optimized benchmark (will be same as baseline if no changes made)
    print_step(6, "Running Benchmark")
    results = run_optimized_benchmark(repo_path, baseline_results)

    # Step 7: Prepare submission
    print_step(7, "Preparing Submission")
    submission = prepare_submission(repo_path, results, github_username)

    print("\n" + "=" * 60)
    print("  STEP 1 COMPLETE!")
    print("=" * 60)
    print(f"""
You now have:
  ✓ Baseline performance measured
  ✓ Optimization opportunities identified
  ✓ Submission payload prepared

Edit the kernels in:
  {chunk_path}

Then re-run this script to measure improvement!
""")


if __name__ == "__main__":
    main()
