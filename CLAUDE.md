# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KernelBench RL is a reinforcement learning training framework for GPU kernel optimization. It fine-tunes LLMs (primarily Qwen-series) to write better CUDA/Triton kernels through multi-turn refinement using the Tinker distributed RL platform and KernelBench evaluation environment.

Key approach:
- Multi-turn RL with structured error feedback (Kevin-32B inspired)
- Optional retrieval-augmented prompting (RA-ICL) from 48K+ kernel corpus
- Thinking tokens (`<think>` blocks) for reasoning before code generation
- Progressive training stages (format → compile → correctness → speed)

## Development Guidelines

**Always use `uv` for Python operations.** Never run Python directly. Use:
- `uv sync` to install dependencies
- `uv run python ...` to execute Python scripts
- `uv run pytest ...` to run tests
- `uv add <package>` to add dependencies

## Common Commands

```bash
# Install dependencies
uv sync

# Training (using justfile)
just train <experiment_name>              # Default: Kevin mode (RA-ICL optional)
just train-kevin <experiment_name>        # Kevin mode explicitly
just train-raicl <experiment_name>        # RA-ICL only (single-turn)
just train-config <config.yaml> <name>    # Custom config

# Manual training
uv run python -m kernel_rl.scripts.train_kernel_rl \
    --config kernel_rl/config/rl_kernelbench.yaml \
    log_path=./runs/<experiment_name>

# Monitoring
just watch-metrics <experiment_name>      # Live metric updates
just logs <experiment_name>               # Tail training logs
just tensorboard <experiment_name>        # Launch TensorBoard
just status                               # Check running jobs

# Run management
just resume <experiment_name>             # Resume from checkpoint
just stop <experiment_name>               # Stop training
just list                                 # List all runs

# RAG index building
just build-rag-index                      # Build full index
just build-rag-index-triton               # Triton kernels only
just build-rag-index-cuda                 # CUDA kernels only

# Evaluation
uv run python -m kernel_rl.scripts.eval_kernel_rl \
    checkpoint_path=./runs/<experiment_name> \
    level=1 \
    output_path=./eval_results.json

# Linting
uv run black kernel_rl/
uv run isort kernel_rl/
uv run mypy kernel_rl/
```

## Architecture

```
kernel_rl/
├── envs/                    # RL environments wrapping KernelBench
│   ├── kernelbench_client.py    # KernelBench evaluation (compile, test, benchmark)
│   ├── kernelbench_env.py       # Single-turn RL environment
│   └── multiturn_kernelbench_env.py  # Kevin-mode multi-turn with error feedback
│
├── rag/                     # Retrieval-Augmented In-Context Learning
│   ├── corpus.py               # KernelCorpus loader (KernelBook + Sakana datasets)
│   ├── retriever.py            # FAISS index + sentence-transformers embeddings
│   └── prompt_builder.py       # RA-ICL prompt construction
│
├── training/                # Core training logic
│   ├── loop.py                 # GRPO-style training loop (main entry)
│   ├── reward.py               # Reward components: format, compile, correctness, speed, thinking
│   ├── models.py               # Model config & Tinker client initialization
│   └── tensorboard_logger.py   # Training metrics visualization
│
├── scripts/                 # CLI entry points
│   ├── train_kernel_rl.py      # Training CLI (loads YAML + CLI overrides)
│   ├── eval_kernel_rl.py       # Evaluation CLI
│   └── build_rag_index.py      # RAG index builder
│
└── config/                  # YAML configurations
    ├── rl_kernelbench.yaml         # Default (Kevin; RA-ICL optional)
    ├── rl_kernelbench_kevin.yaml   # Kevin-mode specific
    └── rl_kernelbench_raicl.yaml   # RA-ICL only
```

## Key Patterns

### Structured Output Format
Models produce outputs in this format:
```
<think>
- Optimization strategy
- Key implementation details
</think>

<KERNEL>
```cuda
// kernel code
```
</KERNEL>
```

### Multi-Turn Flow (Kevin Mode)
Each problem allows multiple refinement turns with error feedback:
1. Turn 1: Problem (optionally with RA-ICL examples) → Kernel_1 → Evaluation
2. Turn 2: Problem + Previous thinking + Error feedback → Kernel_2 → ...
3. Continue until success or max_turns reached
4. Discounted returns: R_t = s_t + γ*s_{t+1} + γ²*s_{t+2} + ...

### Reward Configuration
Weights are configurable per training stage:
- `format_reward_weight`: Valid `<KERNEL>` block extraction
- `compile_reward_weight`: Successful compilation
- `correctness_reward_weight`: Test passage (supports partial credit)
- `speed_reward_weight`: Speedup over baseline (enable `measure_performance`)
- `thinking_reward_weight`: Using `<think>` blocks

### Configuration System
Uses `chz` framework merging:
1. YAML config file (base defaults)
2. CLI arguments with dot notation override (e.g., `dataset_builder.batch_size=4`)
3. Environment variables (for secrets like `TINKER_API_KEY`)

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | Yes | Tinker distributed training API key |
| `KERNELBENCH_ROOT` | No | Path to KernelBench repo (auto-detected) |

## Output Structure

Training runs output to `./runs/<experiment_name>/`:
- `logs.log`: Training logs
- `metrics.jsonl`: Per-batch metrics (JSON lines)
- `checkpoints.jsonl`: Checkpoint paths for resume
- `tensorboard/`: TensorBoard event files


