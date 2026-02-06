# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QUASAR is a Bittensor subnet (netuid 439 mainnet / 24 testnet) where miners compete to write the fastest Triton GPU kernels for QuasarAttention. Miners optimize kernels in a forked `flash-linear-attention` repo, validators benchmark submissions in sandboxed Docker containers, and a FastAPI server coordinates everything.

Scoring: `weighted_score = tokens_per_sec × league_multiplier` (league multipliers range from 0.5x at 100k tokens to 3.0x at 1M+). Top 4 miners share rewards (60/25/10/5%). Logit verification against `Qwen/Qwen2.5-0.5B-Instruct` is required (cosine_sim ≥ 0.99, max_diff ≤ 0.1).

## Common Commands

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run miner (requires GITHUB_TOKEN, VALIDATOR_API_URL env vars)
python neurons/miner.py --wallet.name miner --wallet.hotkey default

# Run validator
python neurons/validator.py --netuid 24 --subtensor.network finney \
  --wallet.name validator --wallet.hotkey default

# Run validator in mock mode (local testing, no blockchain)
python neurons/validator.py --mock --logging.debug

# Run tests (requires GPU + fla package installed)
python -m pytest tests/test_quasar_mining.py -v
python -m pytest tests/test_quasar_mining.py::test_quasar_basic -v

# Lint
black --line-length 79 --check .
pylint --fail-on=W,E,F .

# Docker
docker-compose up --build

# Local kernel validation (test before submitting)
python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --seq-len 4096
python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --seq-len 1000000
python scripts/local_validator.py --repo-path ./quasar_work/flash-linear-attention --full
```

## Architecture

```
Miner (neurons/miner.py)
  ├── Optimizes Triton kernels in quasar_work/flash-linear-attention/fla/ops/quasar/
  ├── Benchmarks throughput at target sequence lengths
  └── Submits results + git commit hash to Validator API

Validator API (validator_api/app.py)
  ├── FastAPI + SQLAlchemy (SQLite/PostgreSQL)
  ├── Tracks submissions, IP rate-limiting, commit-reveal anti-cheat
  └── Serves /submit_submission, /get_submission_stats, /leaderboard

Validator (neurons/validator.py)
  ├── Polls API for pending submissions
  ├── Clones miner repos, validates imports, benchmarks in Docker
  ├── Runs logit verification (inference_verification.py)
  └── Sets weights on Bittensor chain

Challenge Runner (challenge/code_runner.py)
  └── Ephemeral Docker containers: no network, 30s timeout, blocked imports
```

### Kernel Optimization Target

The core optimization target is `quasar_work/flash-linear-attention/fla/ops/quasar/`:

| File | Purpose |
|------|---------|
| `chunk.py` | Main chunk-wise forward pass (primary optimization target) |
| `forward_substitution.py` | Triangular solve kernel |
| `chunk_intra_token_parallel.py` | Intra-token parallel variant |
| `fused_recurrent.py` | Fused recurrent implementation |
| `gate.py` | Gating operations |

The `QuasarAttention` layer (`fla/layers/quasar.py`) calls into these ops. The validator benchmarks through `QuasarAttention(hidden_size=512, head_dim=64, num_heads=8, mode="chunk")`.

### Scoring & Leagues

```
weighted_score = tokens_per_sec × league_multiplier

100k: 0.5x  |  200k: 0.75x  |  300k: 1.0x   |  400k: 1.25x  |  500k: 1.5x
600k: 1.75x |  700k: 2.0x   |  800k: 2.25x  |  900k: 2.5x   |  1M+: 3.0x
```

### Core Library (`quasar/`)

- `protocol.py` — Synapse definitions: InfiniteContextSynapse, CommitRevealData, BenchmarkTaskInfo, InferenceVerificationSynapse
- `base/` — BaseMinerNeuron, BaseValidatorNeuron (Bittensor lifecycle)
- `validator/reward.py` — Reward calculation with scoring weights from `subnet_config.json`
- `benchmarks/` — Loaders for LongBench, HotpotQA, GovReport, Needle-in-Haystack
- `inference_verification.py` — Logit comparison against reference model

## Configuration

- `subnet_config.json` — Scoring weights (memory_retention 35%, position_understanding 25%, coherence 20%, tokens_per_sec 10%, scaling_efficiency 10%), evaluation cycle 90s
- `hfa_config.json` — Model params, max context 100k tokens, checkpoint intervals
- `.env` — Secrets: GITHUB_TOKEN, WALLET_MINER_NAME, WALLET_HOTKEY, NETUID, VALIDATOR_API_URL
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for GPU memory management

## Key Patterns

- The `q.view(B, H, NT, BT, S)` reshape from `[B, T, H, S]` in `chunk.py` interleaves H and T dims — this is the existing convention throughout the codebase, do not change it
- Triton kernels use `tl.constexpr` for block sizes; `tl.dot` requires power-of-2 inner dims ≥ 16
- Triton autotune `key` params must exactly match kernel argument names
- Reference patterns for kernel optimization live in `fla/ops/common/chunk_delta_h.py` (state recurrence) and `fla/ops/delta_rule/wy_fast.py` (fused W+U)
- Tests require GPU; test framework includes OOM retry with parameter degradation
- Forbidden imports in kernel files: `fla.ops.gla`, `fla.ops.kda`
