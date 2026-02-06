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
    "import torch.nn.functional as F",
    "import triton",
    "import triton.language as tl",
    "from fla.ops.utils.index import prepare_chunk_indices",
    "from fla.ops.quasar.forward_substitution import forward_substitution_kernel",
    "from fla.utils import IS_AMD",
    "from fla.utils import autocast_custom_bwd",
    "from fla.utils import autocast_custom_fwd",
    "from fla.utils import autotune_cache_kwargs",
    "from fla.utils import check_shared_mem",
    "from fla.utils import input_guard",
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
    target_files = [
        "chunk.py",
        "chunk_intra_token_parallel.py",
        "forward_substitution.py",
        "fused_recurrent.py",
        "gate.py",
    ]

    errors = []

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
                    errors.append(f"{filename}: Missing required import: {required}")

    return len(errors) == 0, errors


# =============================================================================
# LEAGUE MULTIPLIERS (matches real validator scoring from validator_api/app.py)
# =============================================================================

LEAGUE_CONFIG = {
    "1M":   {"min_seq": 1_000_000, "multiplier": 3.0},
    "900k": {"min_seq": 900_000,   "multiplier": 2.5},
    "800k": {"min_seq": 800_000,   "multiplier": 2.25},
    "700k": {"min_seq": 700_000,   "multiplier": 2.0},
    "600k": {"min_seq": 600_000,   "multiplier": 1.75},
    "500k": {"min_seq": 500_000,   "multiplier": 1.5},
    "400k": {"min_seq": 400_000,   "multiplier": 1.25},
    "300k": {"min_seq": 300_000,   "multiplier": 1.0},
    "200k": {"min_seq": 200_000,   "multiplier": 0.75},
    "100k": {"min_seq": 100_000,   "multiplier": 0.5},
}

def get_league(seq_len: int) -> Tuple[str, float]:
    """Get league name and multiplier for a sequence length."""
    for league_name, config in LEAGUE_CONFIG.items():
        if seq_len >= config["min_seq"]:
            return league_name, config["multiplier"]
    return "100k", 0.5


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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    model(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
    """Check for NaN/Inf and output consistency using production thresholds.

    Production thresholds (from quasar/inference_verification.py):
        cosine_sim >= 0.99
        max_diff <= 0.1
    """
    import torch
    import numpy as np

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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    out = model(x)
                    # Handle tuple return (output, state)
                    if isinstance(out, tuple):
                        out = out[0]
                    outputs.append(out.clone())

        # Check for NaN/Inf
        has_nan = any(torch.isnan(o).any().item() for o in outputs)
        has_inf = any(torch.isinf(o).any().item() for o in outputs)

        # Check consistency using production thresholds
        # Flatten outputs to 1-D for cosine similarity (matching verify_logits logic)
        ref = outputs[0].float().cpu().numpy().flatten()
        cosine_sims = []
        max_diffs = []
        for i in range(1, len(outputs)):
            cmp = outputs[i].float().cpu().numpy().flatten()
            # Cosine similarity
            norm_ref = np.linalg.norm(ref)
            norm_cmp = np.linalg.norm(cmp)
            if norm_ref < 1e-9 or norm_cmp < 1e-9:
                cosine_sims.append(0.0)
            else:
                cosine_sims.append(float(np.dot(ref, cmp) / (norm_ref * norm_cmp)))
            # Max absolute difference
            max_diffs.append(float(np.max(np.abs(ref - cmp))))

        # Production thresholds: cosine_sim >= 0.99, max_diff <= 0.1
        cosine_ok = all(cs >= 0.99 for cs in cosine_sims)
        diff_ok = all(d <= 0.1 for d in max_diffs)
        is_consistent = cosine_ok and diff_ok

        del model, x, outputs
        torch.cuda.empty_cache()

        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_consistent": is_consistent,
            "cosine_sims": cosine_sims,
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

def validate_submission(repo_path: str, target_seq_len: int, full_benchmark: bool = False,
                        claimed_tps: float = None) -> Dict:
    """
    Validate a submission following the real validator's process.

    Args:
        repo_path: Path to the flash-linear-attention repo.
        target_seq_len: Target sequence length for benchmarking.
        full_benchmark: If True, test multiple sequence lengths.
        claimed_tps: Optional claimed tokens/sec for production tolerance scoring.

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
        if correctness.get("cosine_sims"):
            print(f"    Cosine similarity: {min(correctness['cosine_sims']):.6f} (threshold: >= 0.99)")
        if correctness.get("max_diffs"):
            print(f"    Max abs diff:      {max(correctness['max_diffs']):.6f} (threshold: <= 0.1)")
    else:
        if correctness.get("error"):
            print(f"  ✗ Error: {correctness['error']}")
        else:
            print(f"  ✗ NaN: {correctness['has_nan']}, Inf: {correctness['has_inf']}")
            print(f"  ✗ Consistent: {correctness['is_consistent']}")
            if correctness.get("cosine_sims"):
                print(f"    Cosine similarity: {min(correctness['cosine_sims']):.6f} (threshold: >= 0.99)")
            if correctness.get("max_diffs"):
                print(f"    Max abs diff:      {max(correctness['max_diffs']):.6f} (threshold: <= 0.1)")
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

        # Production tolerance scoring (when --claimed is provided)
        if claimed_tps is not None:
            print_section("Production Tolerance Scoring")
            if claimed_tps <= 0:
                print(f"  ✗ Invalid claimed value: {claimed_tps} (must be > 0)")
            else:
                tolerance = 0.9  # 90% of claimed
                passed = tokens_per_sec >= claimed_tps * tolerance
                score = (1.0 + (tokens_per_sec - claimed_tps) / claimed_tps) if passed else 0.0
                print(f"  Claimed:  {claimed_tps:,.0f} tok/s")
                print(f"  Actual:   {tokens_per_sec:,.0f} tok/s")
                print(f"  Diff:     {(tokens_per_sec - claimed_tps) / claimed_tps * 100:+.2f}%")
                print(f"  Score:    {score:.4f}")
                if passed:
                    print(f"  ✓ PASS (actual >= claimed * 0.9)")
                else:
                    print(f"  ✗ FAIL (actual < claimed * 0.9)")
                results["final_score"]["claimed_tps"] = claimed_tps
                results["final_score"]["tolerance_score"] = score

        results["overall_valid"] = True
    else:
        results["final_score"] = None
        results["overall_valid"] = False
        print(f"  ✗ Benchmark failed at target sequence length")

    return results


def run_logit_verification(repo_path: str) -> Dict:
    """Smoke-test the logit verification pipeline against the Qwen reference model.

    Requires the ``transformers`` package and downloads the
    ``Qwen/Qwen2.5-0.5B-Instruct`` model on first use.

    This runs the reference model twice with the same prompt and compares the
    captured logits using the production ``verify_logits()`` thresholds.  In
    production the validator compares the *miner's container* output against
    the reference model; locally we can only verify that the pipeline itself
    works and that the reference model produces deterministic logits.

    Returns dict with verification results.
    """
    try:
        from quasar.inference_verification import (
            ReferenceModel,
            generate_verification_challenge,
            verify_logits,
        )
        import asyncio

        print("  Loading reference model (Qwen/Qwen2.5-0.5B-Instruct)...")
        ref_model = ReferenceModel("Qwen/Qwen2.5-0.5B-Instruct")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ref_model.load())

            # Generate challenge — use shorter gen_len for local testing speed
            challenge = generate_verification_challenge(ref_model)
            prompt = challenge["prompt"]
            gen_len = 32
            logits_at_step = max(1, min(challenge["logits_at_step"], gen_len))

            print(f"  Challenge: prompt_len={len(prompt)}, gen_len={gen_len}, capture_step={logits_at_step}")

            # Run reference model inference twice and compare (determinism check)
            print("  Running reference inference (run 1/2)...")
            ref_result = loop.run_until_complete(
                ref_model.inference(prompt, gen_len, logits_at_step)
            )

            if ref_result.get("captured_logits") is None:
                return {
                    "passed": False,
                    "reason": "Reference model failed to capture logits (run 1)",
                }

            print("  Running reference inference (run 2/2)...")
            ref_result_2 = loop.run_until_complete(
                ref_model.inference(prompt, gen_len, logits_at_step)
            )

            if ref_result_2.get("captured_logits") is None:
                return {
                    "passed": False,
                    "reason": "Reference model failed to capture logits (run 2)",
                }
        finally:
            loop.close()

        # Compare logits using production verify_logits
        verification = verify_logits(
            ref_result_2["captured_logits"],
            ref_result["captured_logits"],
        )

        return {
            "passed": verification.verified,
            "cosine_sim": verification.cosine_sim,
            "max_abs_diff": verification.max_abs_diff,
            "reason": verification.reason,
        }

    except ImportError as e:
        return {
            "passed": False,
            "reason": f"Missing dependency: {e}. Install with: pip install transformers",
        }
    except Exception as e:
        return {
            "passed": False,
            "reason": f"Logit verification error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="QUASAR Local Validator")
    parser.add_argument("--repo-path", required=True, help="Path to flash-linear-attention repo")
    parser.add_argument("--seq-len", type=int, default=1000000, help="Target sequence length")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (multiple seq lengths)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--claimed", type=float, default=None,
                        help="Claimed tokens/sec for production tolerance scoring")
    parser.add_argument("--logit-verify", action="store_true",
                        help="Run logit verification against Qwen reference model (requires transformers)")
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
    if args.claimed is not None:
        print(f"Claimed tokens/sec: {args.claimed:,.0f}")
    if args.logit_verify:
        print(f"Logit verification: enabled")

    # Run validation
    results = validate_submission(str(repo_path), args.seq_len, args.full,
                                  claimed_tps=args.claimed)

    # Step 5 (optional): Logit verification
    if args.logit_verify and results.get("overall_valid"):
        print_section("Logit Verification (Qwen Reference Model)")
        logit_result = run_logit_verification(str(repo_path))
        results["logit_verification"] = logit_result

        if logit_result["passed"]:
            print(f"  ✓ PASSED")
            if logit_result.get("cosine_sim") is not None:
                print(f"    Cosine similarity: {logit_result['cosine_sim']:.6f} (threshold: >= 0.99)")
            if logit_result.get("max_abs_diff") is not None:
                print(f"    Max abs diff:      {logit_result['max_abs_diff']:.6f} (threshold: <= 0.1)")
        else:
            print(f"  ✗ FAILED: {logit_result.get('reason', 'unknown')}")
            results["overall_valid"] = False

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
            if fs.get("tolerance_score") is not None:
                print(f"  Production tolerance score: {fs['tolerance_score']:.4f}")
    else:
        print("  ✗ FAILED - Please fix issues before submitting")
        if results.get("import_issues"):
            print("\n  Import issues:")
            for issue in results["import_issues"]:
                print(f"    - {issue}")
        if results.get("logit_verification", {}).get("reason"):
            print(f"\n  Logit verification: {results['logit_verification']['reason']}")


if __name__ == "__main__":
    main()
