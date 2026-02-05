#!/usr/bin/env python3
"""
QUASAR Subnet - Local Validator

Simulates the validator's evaluation process locally.
Tests your optimizations before submitting to the network.

Usage:
    python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention
    python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --seq-len 1000000
    python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --full

Features:
    - Import validation (required/forbidden imports)
    - Performance benchmarking at multiple sequence lengths
    - League multiplier calculation
    - Weighted score computation
    - JSON result output
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_section(title: str):
    print(f"\n[{title}]")
    print("-" * 50)


# =============================================================================
# IMPORT VALIDATION (matches real validator)
# =============================================================================

REQUIRED_IMPORTS = [
    "import torch",
    "import triton",
    "import triton.language as tl",
]

FORBIDDEN_IMPORTS = [
    "from fla.ops.gla",
    "from fla.ops.kda",
    "import fla.ops.gla",
    "import fla.ops.kda",
]

def validate_imports(repo_path: str) -> Tuple[bool, List[str]]:
    """Validate that files have required imports and no forbidden imports."""
    quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
    target_files = ["chunk.py", "forward_substitution.py"]

    errors = []
    warnings = []

    for filename in target_files:
        file_path = os.path.join(quasar_dir, filename)
        if not os.path.exists(file_path):
            errors.append(f"Missing file: {filename}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for forbidden imports
        for forbidden in FORBIDDEN_IMPORTS:
            if forbidden in content:
                errors.append(f"{filename}: Forbidden import found: {forbidden}")

        # Check for required imports (only for chunk.py)
        if filename == "chunk.py":
            for required in REQUIRED_IMPORTS:
                if required not in content:
                    warnings.append(f"{filename}: Missing import: {required}")

    return len(errors) == 0, errors + warnings


# =============================================================================
# LEAGUE MULTIPLIERS (matches real validator scoring)
# =============================================================================

LEAGUE_CONFIG = {
    "2M":    {"min_seq": 2_000_000, "multiplier": 4.0},
    "1.5M":  {"min_seq": 1_500_000, "multiplier": 3.5},
    "1M":    {"min_seq": 1_000_000, "multiplier": 3.0},
    "512K":  {"min_seq": 512_000,   "multiplier": 2.0},
    "124K":  {"min_seq": 124_000,   "multiplier": 1.5},
    "32K":   {"min_seq": 32_000,    "multiplier": 1.0},
}

def get_league(seq_len: int) -> Tuple[str, float]:
    """Get league name and multiplier for a sequence length."""
    for league_name, config in LEAGUE_CONFIG.items():
        if seq_len >= config["min_seq"]:
            return league_name, config["multiplier"]
    return "Mini", 0.5


# =============================================================================
# PERFORMANCE BENCHMARK
# =============================================================================

def run_performance_benchmark(repo_path: str, seq_len: int, num_runs: int = 10) -> Dict:
    """Run performance benchmark at specified sequence length."""
    import torch

    # Add repo to path temporarily
    sys.path.insert(0, repo_path)

    try:
        from fla.layers.quasar import QuasarAttention

        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
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
                with torch.no_grad():
                    model(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        tokens_per_sec = (seq_len * num_runs) / elapsed
        time_ms = (elapsed / num_runs) * 1000
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        vram_gb = vram_mb / 1024

        del model, x
        torch.cuda.empty_cache()

        return {
            "tokens_per_sec": tokens_per_sec,
            "time_ms": time_ms,
            "vram_mb": vram_mb,
            "vram_gb": vram_gb,
            "seq_len": seq_len,
            "num_runs": num_runs,
            "success": True,
        }

    except Exception as e:
        return {
            "tokens_per_sec": 0,
            "time_ms": 0,
            "vram_mb": 0,
            "vram_gb": 0,
            "seq_len": seq_len,
            "num_runs": num_runs,
            "success": False,
            "error": str(e),
        }
    finally:
        # Remove repo from path
        if repo_path in sys.path:
            sys.path.remove(repo_path)


def run_correctness_check(repo_path: str, seq_len: int = 4096) -> Dict:
    """Check for NaN/Inf and output consistency."""
    import torch

    sys.path.insert(0, repo_path)

    try:
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

        # Run multiple times
        outputs = []
        for _ in range(3):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    out = model(x)
                    outputs.append(out.clone())

        # Check for NaN/Inf
        has_nan = any(torch.isnan(o).any().item() for o in outputs)
        has_inf = any(torch.isinf(o).any().item() for o in outputs)

        # Check consistency
        max_diffs = []
        for i in range(1, len(outputs)):
            diff = (outputs[0] - outputs[i]).abs().max().item()
            max_diffs.append(diff)

        is_consistent = all(d < 1e-3 for d in max_diffs)

        del model, x, outputs
        torch.cuda.empty_cache()

        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_consistent": is_consistent,
            "max_diffs": max_diffs,
            "passed": not has_nan and not has_inf and is_consistent,
        }

    except Exception as e:
        return {
            "has_nan": None,
            "has_inf": None,
            "is_consistent": None,
            "passed": False,
            "error": str(e),
        }
    finally:
        if repo_path in sys.path:
            sys.path.remove(repo_path)


# =============================================================================
# MAIN VALIDATION FLOW
# =============================================================================

def validate_submission(repo_path: str, target_seq_len: int, full_benchmark: bool = False) -> Dict:
    """
    Validate a submission following the real validator's process.

    Returns comprehensive validation results.
    """
    import torch

    results = {
        "repo_path": repo_path,
        "target_seq_len": target_seq_len,
        "timestamp": datetime.now().isoformat(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
    }

    # Step 1: Import validation
    print_section("Import Validation")
    imports_valid, import_issues = validate_imports(repo_path)
    results["imports_valid"] = imports_valid
    results["import_issues"] = import_issues

    if imports_valid:
        print("  ✓ All imports valid")
    else:
        print("  ✗ Import validation failed:")
        for issue in import_issues:
            print(f"    - {issue}")
        results["overall_valid"] = False
        return results

    # Step 2: Correctness check
    print_section("Correctness Check")
    correctness = run_correctness_check(repo_path)
    results["correctness"] = correctness

    if correctness["passed"]:
        print("  ✓ No NaN/Inf detected")
        print("  ✓ Output is consistent across runs")
    else:
        if correctness.get("error"):
            print(f"  ✗ Error: {correctness['error']}")
        else:
            print(f"  ✗ NaN: {correctness['has_nan']}, Inf: {correctness['has_inf']}")
            print(f"  ✗ Consistent: {correctness['is_consistent']}")
        results["overall_valid"] = False
        return results

    # Step 3: Performance benchmarks
    print_section("Performance Benchmark")

    if full_benchmark:
        # Test multiple sequence lengths
        seq_lengths = [4096, 16384, 65536, 262144, target_seq_len]
        seq_lengths = sorted(set(seq_lengths))
    else:
        seq_lengths = [target_seq_len]

    benchmarks = {}
    for seq_len in seq_lengths:
        print(f"\n  Testing seq_len={seq_len:,}...")
        bench = run_performance_benchmark(repo_path, seq_len)
        benchmarks[seq_len] = bench

        if bench["success"]:
            league, mult = get_league(seq_len)
            weighted = bench["tokens_per_sec"] * mult
            print(f"    Tokens/sec: {bench['tokens_per_sec']:,.0f}")
            print(f"    VRAM: {bench['vram_gb']:.2f} GB")
            print(f"    League: {league} ({mult}x)")
            print(f"    Weighted: {weighted:,.0f}")
        else:
            print(f"    ✗ Failed: {bench.get('error', 'Unknown error')}")

    results["benchmarks"] = benchmarks

    # Step 4: Calculate final score
    print_section("Final Score")

    target_bench = benchmarks.get(target_seq_len, {})
    if target_bench.get("success"):
        tokens_per_sec = target_bench["tokens_per_sec"]
        league, multiplier = get_league(target_seq_len)
        weighted_score = tokens_per_sec * multiplier

        results["final_score"] = {
            "tokens_per_sec": tokens_per_sec,
            "league": league,
            "multiplier": multiplier,
            "weighted_score": weighted_score,
        }

        print(f"  Target sequence length: {target_seq_len:,}")
        print(f"  Tokens/sec: {tokens_per_sec:,.0f}")
        print(f"  League: {league}")
        print(f"  Multiplier: {multiplier}x")
        print(f"  Weighted Score: {weighted_score:,.0f}")

        results["overall_valid"] = True
    else:
        results["final_score"] = None
        results["overall_valid"] = False
        print(f"  ✗ Benchmark failed at target sequence length")

    return results


def main():
    parser = argparse.ArgumentParser(description="QUASAR Local Validator")
    parser.add_argument("--repo-path", required=True, help="Path to flash-linear-attention repo")
    parser.add_argument("--seq-len", type=int, default=1000000, help="Target sequence length")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (multiple seq lengths)")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not (repo_path / "fla/ops/quasar/chunk.py").exists():
        print(f"Error: Not a valid flash-linear-attention repo: {repo_path}")
        sys.exit(1)

    print_header("QUASAR Local Validator")
    print(f"Repository: {repo_path}")
    print(f"Target sequence length: {args.seq_len:,}")
    print(f"Full benchmark: {args.full}")

    # Run validation
    results = validate_submission(str(repo_path), args.seq_len, args.full)

    # Save results
    output_path = args.output or Path(__file__).parent.parent / "local_validator_result.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print_header("Validation Summary")

    if results.get("overall_valid"):
        print("  ✓ PASSED - Ready for submission!")
        if results.get("final_score"):
            fs = results["final_score"]
            print(f"\n  Your score: {fs['weighted_score']:,.0f}")
            print(f"  ({fs['tokens_per_sec']:,.0f} tok/s × {fs['multiplier']}x {fs['league']} multiplier)")
    else:
        print("  ✗ FAILED - Please fix issues before submitting")
        if results.get("import_issues"):
            print("\n  Import issues:")
            for issue in results["import_issues"]:
                print(f"    - {issue}")


if __name__ == "__main__":
    main()
