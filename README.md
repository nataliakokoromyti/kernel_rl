# KernelBench RL

RL training for GPU kernel optimization using [Tinker](https://tinker-docs.thinkingmachines.ai/) and [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

## Overview

This project uses **Reinforcement Learning with Verifiable Rewards (RLVR)** to fine-tune language models to write better CUDA/Triton kernels. The training framework:

- Uses **KernelBench** as the environment and reward source
- Uses **Tinker** for distributed LoRA fine-tuning with GRPO-style RL
- **Default mode**: Kevin-style multi-turn refinement + RA-ICL retrieval
- Supports **Qwen3 thinking tokens** (`<think>...</think>` blocks)
- Supports progressive training stages: format → compile → correctness → speed

## Quick Start

```bash
# 1. Build the RAG index (one-time, ~10 min)
just build-rag-index

# 2. Start training (Kevin mode + RA-ICL is default)
just train my_experiment

# 3. Monitor progress
just watch-kevin my_experiment

# 4. Resume if crashed
just resume my_experiment
```

## Environment

This project is designed to run on a RunPod instance with:

- Image: `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel`
- GPU: NVIDIA L40S (or similar)
- Working directory: `/workspace`

### Directory Structure

```
/workspace/kernel_dev/
├── kernel-rl/           # This project
├── KernelBench/         # KernelBench benchmark
└── tinker-cookbook/     # Tinker cookbook examples
```

## Setup

### 1. Clone repositories

```bash
cd /workspace/kernel_dev
git clone https://github.com/ScalingIntelligence/KernelBench.git
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
# kernel-rl should already be here
```

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

```bash
cd /workspace/kernel_dev/kernel-rl
uv sync
```

### 4. Configure environment variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and set your Tinker API key (get it from https://console.tinker.thinkingmachines.ai):

```bash
TINKER_API_KEY=your-api-key-here
```

The `.env` file is automatically loaded when running scripts.

## RA-ICL (Retrieval-Augmented In-Context Learning)

This project supports **RA-ICL** prompting, which retrieves similar kernel examples from large corpora to provide relevant in-context examples for each problem.

### Data Sources

| Dataset | Size | Backend | Description |
|---------|------|---------|-------------|
| [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | 18.2K | Triton | PyTorch → Triton via Inductor |
| [AI-CUDA-Engineer](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) | 30.6K | CUDA | Sakana AI's CUDA kernel archive |

### Quick Start

```bash
# 1. Build the RAG index (one-time, ~10 min)
just build-rag-index

# 2. Train with RA-ICL prompts
just train-raicl run=my_raicl_run
```

### Manual Commands

```bash
# Build index (both datasets)
uv run python -m kernel_rl.scripts.build_rag_index --output ./kernel_rag_index

# Build Triton-only index (KernelBook)
uv run python -m kernel_rl.scripts.build_rag_index --output ./kernel_rag_index --triton-only

# Build CUDA-only index (Sakana)
uv run python -m kernel_rl.scripts.build_rag_index --output ./kernel_rag_index --cuda-only

# Train with RA-ICL
uv run python -m kernel_rl.scripts.train_kernel_rl \
    --config kernel_rl/config/rl_kernelbench_raicl.yaml \
    log_path=./runs/raicl_experiment
```

### Configuration

Add these options to your config:

```yaml
dataset_builder:
  prompt_option: "raicl"              # Enable RA-ICL
  rag_index_path: "./kernel_rag_index"  # Path to index
  raicl_k: 3                          # Examples per prompt
```

### How It Works

```
Query PyTorch Code → BGE-Code Embedding → FAISS Search → Top-K Examples → Inject into Prompt
```

1. **Embedding**: PyTorch code is embedded using BGE-Code
2. **Retrieval**: FAISS finds most similar examples from the corpus
3. **Prompting**: Retrieved examples are injected as few-shot demonstrations
4. **Generation**: Model generates kernel with relevant context

### Index Build Options

| Option | Description |
|--------|-------------|
| `--output` | Output directory for index |
| `--triton-only` | Only KernelBook (Triton) examples |
| `--cuda-only` | Only Sakana (CUDA) examples |
| `--model` | Embedding model (default: `BAAI/bge-code-v1`) |
| `--sakana-levels` | Sakana levels to include (default: `1,2,3`) |
| `--include-incorrect` | Include incorrect kernels from Sakana |

**Recommended model:** For better code similarity matching, use `nomic-ai/nomic-embed-code`:

```bash
just build-rag-index --model nomic-ai/nomic-embed-code
```

## Structured Output Format

The model is trained to produce outputs in a structured format with explicit reasoning:

    <think>
    1-5 short bullet points describing:
    - What optimization strategy you will use
    - Key implementation details (tiling, memory layout, etc.)
    - Any constraints or edge cases to handle

    Keep this section under 150 tokens.
    </think>

    <KERNEL>
    ```python
    class ModelNew(nn.Module):
        ...
    ```
    </KERNEL>

### Thinking Tokens

This project supports **Qwen3's native thinking tokens** (`<think>...</think>`) following the approach from [Kevin-32B](https://cognition.ai/blog/kevin-32b):

- **Structured reasoning**: The `<think>` block encourages the model to plan before coding
- **Multi-turn context**: In refinement turns, the previous turn's thinking summary is included to preserve reasoning state
- **Thinking reward**: A small bonus rewards using the thinking format (configurable via `thinking_weight`)

Based on research findings (Kevin paper), the thinking reward is **generous with no harsh penalties**:
- No thinking: neutral reward (0.5)
- Any thinking: full bonus (1.0)
- Very long thinking: gentle decay to floor (0.7)

This avoids the "response collapse" that occurs with harsh length penalties.

### Parsing

The system parses both Qwen3 (`<think>`) and Kevin (`<THOUGHT>`) formats. If the structured format isn't found, it falls back to extracting Python code blocks.

## Training

The default configuration uses **Kevin mode (multi-turn) + RA-ICL**. This means:
- Model gets 8 refinement attempts per problem
- Each attempt receives error feedback from the previous attempt
- RA-ICL provides relevant kernel examples from a 34K+ corpus
- Checkpoints saved after every batch for crash recovery

### Using Justfile Commands

```bash
# Start training (uses default config with Kevin + RA-ICL)
just train my_experiment

# Monitor Kevin mode metrics
just watch-kevin my_experiment

# Resume from checkpoint if crashed
just resume my_experiment

# View logs
just logs my_experiment

# Check training status
just status
```

### Manual Training

```bash
uv run python -m kernel_rl.scripts.train_kernel_rl \
    --config kernel_rl/config/rl_kernelbench.yaml \
    log_path=./runs/my_experiment
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | HuggingFace model ID | `Qwen/Qwen3-30B-A3B` |
| `mode` | `"single_turn"` or `"multi_turn"` | `"multi_turn"` |
| `max_turns` | Refinement attempts (Kevin mode) | `4` |
| `gamma` | Discount factor for future rewards | `0.4` |
| `level` | KernelBench level (1-4) | `1` |
| `batch_size` | Problems per batch | `2` |
| `group_size` | Rollouts per problem | `4` |
| `learning_rate` | LoRA learning rate | `1e-4` |
| `lora_rank` | LoRA rank | `32` |
| `max_tokens` | Max generation tokens | `4096` |
| `save_every` | Checkpoint frequency | `1` (every batch) |

### Checkpoints and Resume

Checkpoints are saved to Tinker cloud after every batch. The checkpoint paths are recorded in `{log_path}/checkpoints.jsonl`.

```bash
# Resume training after a crash
just resume my_experiment

# Or manually:
uv run python -m kernel_rl.scripts.train_kernel_rl \
    --config kernel_rl/config/rl_kernelbench.yaml \
    log_path=./runs/my_experiment \
    load_checkpoint_path=./runs/my_experiment
```

**Note**: Kernel evaluation can sometimes crash the GPU with illegal memory access errors. The frequent checkpointing ensures minimal progress loss.

### Reward Configuration

The reward is a weighted combination of components:

| Weight | Description | Default |
|--------|-------------|---------|
| `reward_format_weight` | Valid `<KERNEL>` block extraction | 0.1 |
| `reward_compile_weight` | Successful compilation | 0.2 |
| `reward_correctness_weight` | Passing tests (partial credit) | 1.0 |
| `reward_speed_weight` | Speedup over baseline | 0.0 |
| `reward_length_weight` | Code brevity (tie-breaking) | 0.05 |
| `reward_thinking_weight` | Using `<think>` blocks | 0.1 |

### Reward Stages

The reward function has presets for different training stages:

1. **Stage 1** (format + compile focus):
   ```bash
   dataset_builder.reward_format_weight=0.3 \
   dataset_builder.reward_compile_weight=0.5 \
   dataset_builder.reward_correctness_weight=0.2
   ```

2. **Stage 2** (correctness focus):
   ```bash
   dataset_builder.reward_format_weight=0.1 \
   dataset_builder.reward_compile_weight=0.2 \
   dataset_builder.reward_correctness_weight=0.7
   ```

3. **Stage 3** (speed optimization):
   ```bash
   dataset_builder.reward_correctness_weight=0.5 \
   dataset_builder.reward_speed_weight=0.35 \
   dataset_builder.measure_performance=true
   ```

## TensorBoard Visualization

Training progress can be monitored in real-time using TensorBoard.

### Launch TensorBoard

```bash
# For a specific run
uv run tensorboard --logdir ./runs/kernel_rl_v1/tensorboard --port 6006

# For all runs
uv run tensorboard --logdir ./runs --port 6006
```

Then open http://localhost:6006 in your browser.

### Available Metrics

| Category | Metrics | Description |
|----------|---------|-------------|
| **Reward** | Mean, StdDev, Min, Max | Reward distribution across trajectories |
| **Kernel Quality** | FormatRate, CompileRate, CorrectRate, CheatRate | Success rates at each stage |
| **Thinking** | thought_length, has_thought | Thinking token usage (chars, boolean) |
| **Progress** | CompletionFraction, LearningRate | Training progress tracking |
| **Timing** | Total, Rollout, Train, SaveCheckpoint | Time breakdown per batch |
| **Per-Level** | RewardMean, CorrectRate per level | Breakdown by difficulty level |
| **Distributions** | Rewards, Speedups, Advantages | Histograms (logged every N batches) |

### Evaluation Metrics

Evaluation metrics can also be logged to TensorBoard alongside training:

```bash
uv run python -m kernel_rl.scripts.eval_kernel_rl \
    checkpoint_path=./runs/kernel_rl_v1/checkpoints/final \
    level=1 \
    output_path=./runs/kernel_rl_v1/eval_results.json \
    tensorboard_log_dir=./runs/kernel_rl_v1 \
    tensorboard_step=100
```

This logs pass@k, speedup metrics, and quality rates to the same TensorBoard.

### TensorBoard Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tensorboard_enabled` | Enable TensorBoard logging | `True` |
| `tensorboard_log_histograms_every` | Log histograms every N batches | `5` |
| `tensorboard_log_per_level` | Log per-level breakdowns | `True` |

## Evaluation

### Evaluate a Checkpoint

```bash
uv run python -m kernel_rl.scripts.eval_kernel_rl \
    checkpoint_path=./runs/kernel_rl_v1/checkpoints/final \
    level=1 \
    output_path=./runs/kernel_rl_v1/eval_results.json
```

### Evaluate Base Model

```bash
uv run python -m kernel_rl.scripts.eval_kernel_rl \
    model_name=Qwen/Qwen2.5-Coder-7B-Instruct \
    level=1 \
    output_path=./baseline_results.json
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TINKER_API_KEY` | Yes | API key from https://console.tinker.thinkingmachines.ai |
| `KERNELBENCH_ROOT` | No | Path to KernelBench repo (auto-detected) |
| `CUDA_VISIBLE_DEVICES` | No | GPU selection (default: all available) |

## KernelBench Compatibility

The existing KernelBench scripts still work as-is:

```bash
# Generate samples with external model
cd /workspace/kernel_dev/KernelBench
uv run python scripts/generate_samples.py \
    dataset_src=huggingface \
    level=1 \
    run_name=my_run \
    server_type=together \
    model_name='together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8'

# Evaluate those generations
uv run python scripts/eval_from_generations.py \
    dataset_src=huggingface \
    level=1 \
    run_name=my_run \
    eval_mode=local
```

## Architecture

```
kernel_rl/
├── env.py                          # Environment variable loading
├── envs/
│   ├── kernelbench_client.py       # KernelBench Python API wrapper
│   ├── kernelbench_env.py          # Single-turn RL environment
│   └── multiturn_kernelbench_env.py # Multi-turn RL environment (Kevin mode)
├── rag/                            # RA-ICL module
│   ├── corpus.py                   # KernelBook + Sakana loaders
│   ├── retriever.py                # FAISS index + embeddings
│   └── prompt_builder.py           # RA-ICL prompt construction
├── training/
│   ├── models.py                   # Model configuration
│   ├── reward.py                   # Reward shaping + discounted returns
│   ├── loop.py                     # GRPO training loop (single + multi-turn)
│   └── tensorboard_logger.py       # TensorBoard visualization
├── evaluation/
│   └── eval_kernelbench.py         # Evaluation utilities
├── scripts/
│   ├── train_kernel_rl.py          # Training CLI
│   ├── eval_kernel_rl.py           # Evaluation CLI
│   └── build_rag_index.py          # RAG index builder
└── config/
    ├── rl_kernelbench.yaml         # Default config (Kevin + RA-ICL)
    ├── rl_kernelbench_raicl.yaml   # RA-ICL config (single-turn)
    └── rl_kernelbench_kevin.yaml   # Kevin mode config (multi-turn, legacy)
```

## Multi-Turn Training (Kevin Mode)

This implementation includes **Kevin-style multi-turn refinement training**, inspired by [Cognition's Kevin-32B](https://cognition.ai/blog/kevin-32b). **This is now the default training mode.**

### How It Works

Instead of generating one kernel per problem, the model gets **T refinement turns** (default T=8):

1. **Turn 0**: Model sees problem + RA-ICL examples → generates first kernel
2. **Turn 1+**: Model sees problem + previous kernel + error feedback → refines
3. Continue until correct or max turns reached

**Rewards use discounted returns** (Kevin paper):
```
R_t = s_t + γ*s_{t+1} + γ²*s_{t+2} + ... (γ = 0.4)
```

This encourages the model to generate kernels that are easy to fix in subsequent turns.

### Quick Start

```bash
# Kevin mode is now the default - just run:
just train my_experiment

# To use single-turn mode instead:
uv run python -m kernel_rl.scripts.train_kernel_rl \
    --config kernel_rl/config/rl_kernelbench.yaml \
    mode=single_turn \
    log_path=./runs/single_turn_experiment
```

### Kevin Mode Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mode` | `"single_turn"` or `"multi_turn"` | `"multi_turn"` |
| `max_turns` | Maximum refinement attempts | `4` |
| `gamma` | Discount factor for future rewards | `0.4` |

### Error Feedback

On each refinement turn, the model receives:
- **Previous thinking summary**: The model's `<think>` content from the last turn (preserves reasoning context)
- **Error category**: COMPILATION ERROR, RUNTIME ERROR, CORRECTNESS ERROR, etc.
- **Detailed error**: Extracted key error message (cleaned of traceback noise)
- **Guidance**: Category-specific hints for fixing the error

Example feedback:

    ## Previous Turn Summary (Model's Analysis)
    - Fusing elementwise ops with matmul
    - Using Triton's tl.dot for the computation
    - Need to handle edge cases for non-divisible dims

    ## Previous Attempt (Turn 1)
    [kernel code shown here]

    ## Evaluation Feedback
    - **Status**: COMPILATION ERROR - Build failed
    - **Compiled**: No
    - **Tests Passed**: 0/5

    ### Error Details
    AttributeError: module 'triton.language' has no attribute 'tanh'

    ## Instructions
    Fix the TRITON syntax/API errors. Check that all kernel functions
    are correctly decorated and all imports are valid.

    Remember: respond using <think>...</think> followed by <KERNEL>...</KERNEL>.

### Kevin Mode Metrics

| Metric | Description |
|--------|-------------|
| `multiturn/success_rate` | Fraction of trajectories that solved the problem |
| `multiturn/avg_turns` | Average turns used (lower = faster solving) |
| `multiturn/compile_rate` | Compile success across all turns |
| `multiturn/correct_rate` | Correctness across all turns |
| `multiturn/max_correct_per_trajectory` | Best correctness achieved per trajectory |
| `multiturn/turn_N/compile_rate` | Per-turn compile rates |

## Future Work

### Multi-Agent Training
The Tinker EnvGroupBuilder pattern supports:
- Solver/Verifier/Corrector agents
- Self-play and adversarial training
- Preference-based reward models

### Curriculum Learning
Progressive training through:
- Level 1 → Level 2 → Level 3 problems
- Increasing correctness requirements
- Adding speed optimization

## Troubleshooting

### CUDA illegal memory access / Training crashes
Generated kernels can sometimes corrupt GPU memory. This is handled by:
- Checkpoints saved after every batch (`save_every: 1`)
- Resume capability: `just resume my_experiment`

If crashes are frequent:
```bash
# Clear GPU memory and restart
nvidia-smi --gpu-reset  # If needed
just resume my_experiment
```

### CUDA out of memory
Reduce `batch_size` or `group_size`:
```bash
batch_size=2 group_size=2
```

### KernelBench import errors
Check `KERNELBENCH_ROOT`:
```bash
export KERNELBENCH_ROOT=/workspace/kernel_dev/KernelBench
```

### Tinker API errors
1. Check your API key is set: `echo $TINKER_API_KEY`
2. Get a key from https://console.tinker.thinkingmachines.ai
3. Check Tinker service status

### Resume not working
Ensure `checkpoints.jsonl` exists in the run directory:
```bash
cat ./runs/my_experiment/checkpoints.jsonl
```
If empty or missing, training crashed before the first checkpoint was saved.

## References

- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- [Kevin-32B](https://cognition.ai/blog/kevin-32b) - Multi-turn kernel RL (Cognition)
- [Kevin-32B Paper](https://arxiv.org/abs/2507.11948) - Details on thinking rewards and length penalties
- [Demystifying Long CoT](https://arxiv.org/abs/2502.03373) - Length-scaling reward research
