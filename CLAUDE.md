# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QUASAR is a Bittensor subnet (netuid 439/24) for evaluating long-context language models (32k to 2M tokens). Miners run long-context models and respond to benchmark evaluation requests; validators assess performance using LongBench, HotpotQA, GovReport, and Needle-in-Haystack datasets.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run miner
python neurons/miner.py \
  --wallet.name miner --wallet.hotkey default \
  --subtensor.network finney --netuid 439 \
  --axon.port 8091 --miner.model_name "silx-ai/Quasar-2M-Base"

# Run miner with agent-based kernel optimization
export GITHUB_TOKEN="your_token"
export VALIDATOR_API_URL="https://quasar-subnet.onrender.com"
export AGENT_ITERATIONS=100
python neurons/miner.py --wallet.name miner --wallet.hotkey default

# Run validator
python neurons/validator.py \
  --netuid 24 --subtensor.network finney \
  --wallet.name validator --wallet.hotkey default \
  --neuron.polling_interval 300

# Run tests (requires GPU and fla package)
python -m pytest tests/test_quasar_mining.py -v

# Run a single test
python -m pytest tests/test_quasar_mining.py::test_quasar_basic -v

# Linting
black --line-length 79 --check .
pylint --fail-on=W,E,F .

# Docker deployment
docker-compose up --build

# Mock mode for local testing
python neurons/validator.py --mock --logging.debug
```

## Architecture

### Core Components

1. **Miners** (`neurons/miner.py`): Load long-context models, process evaluation requests, stream responses to validators

2. **Validators** (`neurons/validator.py`): Poll API for submissions, create Docker containers for sandboxed code execution, evaluate against test cases, update scores

3. **Validator API** (`validator_api/app.py`): FastAPI REST API with SQLAlchemy ORM for submission management, rate limiting, CORS support

4. **Challenge Runner** (`challenge/server.py`, `challenge/code_runner.py`): Docker-based sandboxed code execution with ephemeral python:3.11-slim containers

5. **Core Library** (`quasar/`):
   - `protocol.py`: Synapse definitions (InfiniteContextSynapse, CommitRevealData, BenchmarkTaskInfo)
   - `benchmarks/`: Benchmark loaders for LongBench, HotpotQA, GovReport, Needle-in-Haystack
   - `base/`: BaseMinerNeuron, BaseValidatorNeuron
   - `validator/`: Reward calculation, scoring, diversity tracking
   - `inference_verification.py`: Logit verification for miner submissions

### Reward Calculation

```python
accuracy = metric_fn(response, expected_answer)  # F1, EM, or ROUGE
multiplier = {32k: 1.0, 124k: 1.2, 512k: 1.5, 1.5m: 1.8, 2m: 2.0}
reward = min(accuracy * multiplier, 1.0)
```

### Scoring Weights (from subnet_config.json)

- Memory retention: 35%, Position understanding: 25%, Coherence: 20%, Tokens/sec: 10%, Scaling efficiency: 10%
- Context multipliers: 32k=1.0x, 124k=1.2x, 512k=1.5x, 1.5M=1.8x, 2M=2.0x
- Exponential bonus for memory retention > 0.8 (score^2)

### Docker Container Flow

Validator creates ephemeral containers per submission:
1. Start python:3.11-slim container with code_runner.py mounted
2. Health check, then execute all test cases sequentially
3. Destroy container after completion

Security: no network access, no write access, dangerous imports blocked, 30-second timeout.

## Key Files

- `quasar/protocol.py`: Defines InfiniteContextSynapse for miner-validator communication
- `neurons/miner.py`: Miner with agent-based kernel optimization
- `neurons/validator.py`: Validator with Docker container evaluation
- `validator_api/app.py`: REST API server
- `challenge/code_runner.py`: Sandboxed execution handler
- `subnet_config.json`: Scoring weights and benchmark configs

## Tech Stack

- **Core**: Python 3.9+, Bittensor 7.0+, PyTorch 2.0+
- **LLM**: Transformers 4.30+, supports Qwen, Kimi, Quasar models
- **Web**: FastAPI, Uvicorn, SQLAlchemy (SQLite/PostgreSQL)
- **Evaluation**: ROUGE, Jieba, FuzzyWuzzy for metrics
- **Monitoring**: WandB integration

## Configuration

- `subnet_config.json`: Scoring weights, context length tests [1000, 5000, 15000, 50000, 100000], evaluation cycle (90s)
- `hfa_config.json`: Model parameters, max context (100k tokens), checkpoint intervals
- GPU memory: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better memory management
