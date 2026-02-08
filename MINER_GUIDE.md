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

## Complete Setup Guide (Example: A100 PCIe 80GB)

### Step 1: Clone QUASAR-SUBNET

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET
cd QUASAR-SUBNET
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### Step 3: Install QUASAR Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
# Required
GITHUB_TOKEN="your_github_token"
GITHUB_USERNAME="your_github_username"

# Miner settings (optional - defaults shown)
VALIDATOR_API_URL="http://localhost:8000"
NETUID=24
SUBTENSOR_NETWORK="test"
WALLET_MINER_NAME="quasar_miner"
WALLET_HOTKEY="default"
TARGET_SEQUENCE_LENGTH=100000
AGENT_ITERATIONS=100
OPTIMIZATION_INTERVAL=300

# Inference server settings (optional)
MINER_INFERENCE_PORT=8001
REFERENCE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="cuda"
```

Or export directly:

```bash
export GITHUB_TOKEN="your_github_token"
export GITHUB_USERNAME="your_github_username"
```

### Step 5: Clone flash-linear-attention

```bash
mkdir -p quasar_work
cd quasar_work
git clone https://github.com/troy12x/flash-linear-attention
cd flash-linear-attention
```

### Step 6: Install flash-linear-attention (Same venv)

```bash
# Make sure .venv is still activated
pip install -e .
```

### Step 7: Verify Installation

```bash
python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")'
python -c 'from fla.layers.quasar import QuasarAttention; print("fla installed")'
```

### Step 8: Run Baseline Benchmark

```bash
cd /path/to/QUASAR-SUBNET
python scripts/step1_first_optimization.py --skip-setup
```

### Step 9: Tune Kernels

```bash
python scripts/step2_kernel_tuning.py
```

## Running with Startup Scripts

The project includes startup scripts that handle environment setup automatically. They load `.env`, activate `.venv`, and check prerequisites.

### Start Miner

```bash
./START_MINER.sh
```

This runs `python -m neurons.miner` with settings from `.env`. It checks for `GITHUB_TOKEN`, `GITHUB_USERNAME`, CUDA availability, and validator API connectivity.

### Start Inference Server

```bash
./START_MINER_INFERENCE.sh
```

Runs an inference server on port 8001 (configurable via `MINER_INFERENCE_PORT`). The server loads the reference model (`Qwen/Qwen2.5-0.5B-Instruct` by default) and exposes:
- `POST /inference` - Run inference with logit capture
- `GET /health` - Health check

### Start Validator

```bash
./START_VALIDATOR.sh
```

Runs `python -m neurons.validator` with logit verification and commit-reveal settings from `.env`. Requires the validator API to be running first (`./START_SERVER.sh`).

## A100 PCIe 80GB Configuration

### Optimal Triton Settings

```python
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
num_stages = 3
num_warps = 8
```

### Target Sequence Lengths

| Sequence Length | Expected VRAM | League Multiplier |
|-----------------|---------------|-------------------|
| 500k            | ~40GB         | 1.5x              |
| 750k            | ~55GB         | 2.0x              |
| 1M              | ~70GB         | **3.0x** (Target) |

### Expected Performance

- **Target**: 40,000-60,000 tokens/sec at 1M context
- **Weighted Score**: 120,000-180,000 (with 3.0x multiplier)

## GPU Configuration Reference

| GPU | VRAM | Max Sequence | Recommended Config |
|-----|------|--------------|-------------------|
| RTX 3090 | 24GB | ~200k | BLOCK_M=64, BLOCK_N=64, num_stages=2 |
| RTX 4090 | 24GB | ~250k | BLOCK_M=64, BLOCK_N=128, num_stages=2 |
| A100 PCIe | 80GB | **1M+** | BLOCK_M=128, BLOCK_N=128, num_stages=3 |
| H100 PCIe | 80GB | **1M+** | BLOCK_M=128, BLOCK_N=256, num_stages=4 |

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

## Submission Methods

### Option 1: Direct Submission (via startup script)

```bash
# Configure in .env, then:
./START_MINER.sh
```

Or run manually:

```bash
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091
```

### Option 2: Commit-Reveal (Recommended for IP Protection)

Prevents validators from copying your code before evaluation:

**Phase 1 - Commit:**
```bash
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

## GPU Memory Configuration

Set this environment variable to avoid CUDA OOM fragmentation issues:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

To make it persistent:

```bash
# Project-level (loaded by START_MINER.sh automatically)
echo 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> .env

# System-wide (all terminals)
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
source ~/.bashrc
```

## Local Validation (Test Before Submitting)

Always validate locally before submitting to avoid failed validations and potential IP bans.

```bash
# Quick correctness check (fast, small sequence)
python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --seq-len 4096

# Performance benchmark at 1M (3x league multiplier)
python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --seq-len 1000000

# Full validation across all sequence lengths
python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --full
```

The validator checks:
- **No NaN/Inf** in outputs
- **Consistency** across repeated runs
- **Cosine similarity** >= 0.99 against reference model
- **Max absolute diff** <= 0.1
- **Tokens/sec** throughput and weighted score

## Running the Miner (Process Management)

### Foreground (simplest)

```bash
./START_MINER.sh
# Stop with Ctrl+C
```

### Background with log file

```bash
./START_MINER.sh > miner.log 2>&1 &

# Check logs
tail -f miner.log
```

### Using screen (recommended for SSH sessions)

```bash
# Start a named session
screen -S miner
./START_MINER.sh

# Detach: Ctrl+A then D
# Reattach later:
screen -r miner
```

### Using pm2 (auto-restart on crashes)

```bash
npm install -g pm2

pm2 start ./START_MINER.sh --name quasar-miner
pm2 logs quasar-miner    # check logs
pm2 restart quasar-miner # restart
pm2 stop quasar-miner    # stop
```

## Troubleshooting

### Module Not Found: cryptography

```bash
pip install cryptography
```

### Module Not Found: fla

```bash
# Make sure you're in the .venv
source .venv/bin/activate

# Install flash-linear-attention
cd quasar_work/flash-linear-attention
pip install -e .
```

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

### Code Changes Not Taking Effect

The miner clones your repo from GitHub to `/tmp/flash-linear-attention-miner/`. Local-only changes won't work. You must **commit and push** to your GitHub repo:

```bash
cd quasar_work/flash-linear-attention
git add -A && git commit -m "your message" && git push
```

Then restart the miner.

### Running Two Miners on One Server

Requires **multiple GPUs** and **separate hotkeys**:

```bash
# Miner 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m neurons.miner --wallet.hotkey hotkey-001 ...

# Miner 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m neurons.miner --wallet.hotkey hotkey-002 ...
```

Not practical with a single GPU — both miners would compete for VRAM and compute, lowering both scores.

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
