# QUASAR Subnet Miner Guide

## Overview

QUASAR is a **kernel optimization competition**. Miners compete by optimizing Triton kernels for QuasarAttention to achieve the highest tokens/sec throughput at long sequence lengths.

## How It Works

1. Fork the `flash-linear-attention` repository
2. Optimize Triton kernels for QuasarAttention
3. Run benchmarks to measure tokens/sec
4. Submit optimized code to the validator API
5. Validators clone your repo, verify performance, and run logit verification
6. **Top 4 miners share rewards** (60% / 25% / 10% / 5%)

## Critical Requirements

### 1. Logit Verification (Must Pass)

Validators compare your model's outputs against a reference model:

| Check | Threshold | Fail Result |
|-------|-----------|-------------|
| Cosine similarity | >= 0.99 | **Excluded from rankings** |
| Max absolute diff | <= 0.1 | **Excluded from rankings** |

**Reference model:** `Qwen/Qwen2.5-0.5B-Instruct`

### 2. Performance Validation

- Validators run your code themselves
- Must achieve **>= 90% of claimed performance**
- If actual < claimed × 0.9 → **Score = 0**

### 3. Import Validation

**Required imports** (in `fla/ops/quasar/chunk.py`):

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

**Forbidden imports** (instant failure):

```python
from fla.ops.gla ...
from fla.ops.kda ...
import fla.ops.gla
import fla.ops.kda
```

## Target Files

Located in `fla/ops/quasar/` of the flash-linear-attention repo:

| File | Priority | Description |
|------|----------|-------------|
| `chunk.py` | **1** | Core chunked attention - optimize first |
| `fused_recurrent.py` | **2** | Fused recurrent attention |
| `forward_substitution.py` | **3** | Forward substitution operations |
| `chunk_intra_token_parallel.py` | **4** | Intra-token parallel processing |
| `gate.py` | **5** | Gate mechanism kernels |

## Scoring System

### League Multipliers

Your weighted score = `tokens_per_sec × league_multiplier`

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

**Example:** 50,000 tok/s at 1M context = 150,000 weighted score

### Reward Distribution

**Top 4 miners only** receive rewards:

| Place | Share |
|-------|-------|
| 1st   | **60%** |
| 2nd   | **25%** |
| 3rd   | **10%** |
| 4th   | **5%** |

### Ranking Criteria (In Order)

1. Logit verification must **PASS**
2. Weighted score (descending)
3. Submission timestamp (ascending - **first wins ties**)
4. Submission ID (final tiebreaker)

## Quick Start: First Optimization

Run the Step 1 script to get started immediately:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set your GitHub credentials
export GITHUB_TOKEN="your_token"
export GITHUB_USERNAME="your_username"

# Run the guided optimization script
python scripts/step1_first_optimization.py
```

This script will:
1. Check your environment (Python, CUDA, Triton)
2. Clone the flash-linear-attention repository
3. Run baseline benchmarks
4. Show optimization opportunities
5. Verify correctness
6. Prepare your submission payload

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (16GB+ VRAM recommended)
- GitHub account with API token
- Bittensor wallet with registration on netuid 439

### Environment Variables

```bash
export GITHUB_TOKEN="your_github_token"
export GITHUB_USERNAME="your_github_username"
export VALIDATOR_API_URL="https://quasar-subnet.onrender.com"

# Optional
export AGENT_ITERATIONS=100
export TARGET_SEQUENCE_LENGTH=100000
```

### Installation

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET
cd QUASAR-SUBNET

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running the Miner

```bash
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091
```

## Submission Methods

### Option 1: Direct Submission

Submit your optimization directly:

```bash
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default
```

### Option 2: Commit-Reveal (Recommended for IP Protection)

Prevents validators from copying your code before evaluation:

**Phase 1 - Commit:**
```bash
# Submit hash of your submission
curl -X POST https://quasar-subnet.onrender.com/commit_submission \
  -H "Content-Type: application/json" \
  -d '{
    "miner_hotkey": "your_ss58_address",
    "commitment_hash": "sha256(salt + fork_url)",
    "target_sequence_length": 1000000,
    "signature": "signed_message"
  }'
```

**Wait ~100 blocks (~20 minutes)**

**Phase 2 - Reveal:**
```bash
curl -X POST https://quasar-subnet.onrender.com/reveal_submission \
  -H "Content-Type: application/json" \
  -d '{
    "submission_id": 123,
    "miner_hotkey": "your_ss58_address",
    "commitment_salt": "your_random_salt",
    "fork_url": "https://github.com/you/flash-linear-attention",
    "commit_hash": "abc123...",
    "tokens_per_sec": 50000.0,
    "signature": "signed_message"
  }'
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/submit_kernel` | POST | Direct submission |
| `/commit_submission` | POST | Phase 1: Submit commitment hash |
| `/reveal_submission` | POST | Phase 2: Reveal actual submission |
| `/pending_reveals` | GET | Check pending commit-reveals |
| `/get_submission_stats` | GET | View submission statistics |
| `/get_current_round` | GET | Current competition round info |
| `/health` | GET | API health check |

### Direct Submission Payload

```json
{
  "miner_hotkey": "your_ss58_address",
  "fork_url": "https://github.com/you/flash-linear-attention",
  "commit_hash": "abc123def456...",
  "target_sequence_length": 1000000,
  "tokens_per_sec": 50000.0,
  "vram_mb": 8192.0,
  "benchmarks": {
    "4096": {"tokens_per_sec": 80000, "vram_mb": 2048},
    "16384": {"tokens_per_sec": 65000, "vram_mb": 4096},
    "100000": {"tokens_per_sec": 55000, "vram_mb": 6144},
    "1000000": {"tokens_per_sec": 50000, "vram_mb": 16000}
  },
  "signature": "signed_message_hex"
}
```

## Benchmarking

### Local Benchmark Script

```python
import torch
import time
import json
from fla.layers.quasar import QuasarAttention

def benchmark(seq_len, num_runs=10):
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
    for _ in range(num_runs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (seq_len * num_runs) / elapsed
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return tokens_per_sec, vram_mb

# Test at target sequence lengths
results = {}
for seq_len in [4096, 16384, 65536, 100000, 500000, 1000000]:
    try:
        tps, vram = benchmark(seq_len)
        results[seq_len] = {'tokens_per_sec': tps, 'vram_mb': vram}
        print(f'{seq_len}: {tps:,.0f} tok/s | {vram:.0f} MB')
    except torch.cuda.OutOfMemoryError:
        print(f'{seq_len}: OOM')
        break

print(json.dumps(results, indent=2))
```

### Run Tests

```bash
python -m pytest tests/test_quasar_mining.py -v
```

## Optimization Strategies

### 1. Target High League Multipliers

The math favors longer sequences:
- 50k tok/s at 1M context (3x) = **150k weighted**
- 100k tok/s at 300k context (1x) = **100k weighted**

Optimize for the longest sequence length you can reliably handle.

### 2. Key Kernel Optimizations

| Technique | Expected Gain | Difficulty |
|-----------|---------------|------------|
| Use bfloat16 | 1.5-2x | Easy |
| Tune BLOCK sizes | 10-30% | Easy |
| Coalesced memory access | 20-50% | Medium |
| Kernel fusion | 30-100% | Medium |
| Custom Triton kernels | 2-5x | Hard |

### 3. Submit Early

First submission wins ties. If you achieve the same weighted score as another miner, the earlier submission ranks higher.

### 4. Don't Inflate Claims

Validators verify your claimed performance. If actual < claimed × 0.9, you get zero score. Be conservative in your claims.

## Troubleshooting

### Logit Verification Failed

- Ensure your optimizations don't change numerical output
- Test locally: compare your output against reference implementation
- Use float32 accumulators for numerical stability

### Import Validation Failed

- Check `chunk.py` has all required imports
- Remove any `fla.ops.gla` or `fla.ops.kda` imports

### Performance Below Claimed

- Always run `torch.cuda.synchronize()` before timing
- Use warmup runs before benchmarking
- Test on similar hardware to validators

### Submission Rejected

- Verify signature is correct
- Check API health: `curl https://quasar-subnet.onrender.com/health`
- Ensure commit hash matches your pushed code

### IP Banned

- 5 failed validations from same IP = 24-hour ban
- Ensure your claims are accurate before submitting

## Quick Reference

```
Goal:           Maximize tokens/sec × league_multiplier
Top 4 Win:      60% / 25% / 10% / 5%
Key Files:      chunk.py, fused_recurrent.py
Verification:   Cosine sim >= 0.99, Max diff <= 0.1
Tolerance:      Must achieve >= 90% of claimed performance
Tiebreaker:     First submission wins
API:            https://quasar-subnet.onrender.com
```

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Linear Attention Survey](https://arxiv.org/abs/2402.13891)
- Target repo: https://github.com/troy12x/flash-linear-attention
- Detailed optimization guide: [KERNEL_OPTIMIZATION_STRATEGY.md](./KERNEL_OPTIMIZATION_STRATEGY.md)
