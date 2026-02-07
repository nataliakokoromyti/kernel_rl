# RL Hyperparameters

This project uses `chz` configs, so every field in `kernel_rl.training.loop.TrainingConfig` (and the nested dataset builders) can be overridden from YAML or the CLI. CLI args use dot notation, e.g. `dataset_builder.batch_size=8`. YAML is flattened the same way when passed with `--config path/to/file.yaml`; CLI flags override YAML values.

Built-in presets:
- `kernel_rl/config/rl_kernelbench.yaml`: Kevin-style multi-turn with no thinking rewards (default rec).
- `kernel_rl/config/rl_kernelbench_kevin.yaml`: Closer to the Kevin-32B paper defaults.
- `kernel_rl/config/rl_kernelbench_raicl.yaml`: Single-turn RA-ICL baseline.

## TrainingConfig (top level)
| Key | Default | What it controls |
| --- | --- | --- |
| `model_name` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HF model identifier used by Tinker. |
| `lora_rank` | `32` | LoRA rank when creating the training client. |
| `learning_rate` | `1e-4` | Adam LR passed to Tinker optim step. |
| `max_tokens` | `4096` | Generation limit for each rollout. |
| `temperature` | `1.0` | Sampling temperature for rollouts. |
| `mode` | `"single_turn"` | `single_turn` or `multi_turn` (Kevin-style refinement). |
| `max_turns` | `8` | Max refinement turns in `multi_turn` mode. |
| `gamma` | `0.4` | Discount for multi-turn returns. |
| `num_substeps` | `1` | Optimizer steps per batch (gradient accumulation). |
| `loss_fn` | `"importance_sampling"` | Loss sent to Tinker (`LossFnType`). |
| `kl_penalty_coef` | `0.0` | Optional KL regularization coefficient. |
| `kl_discount_factor` | `0.0` | Discount for KL term. |
| `remove_constant_reward_groups` | `true` | Drop groups where all rewards are equal. |
| `log_path` | `./runs/kernel_rl` | Checkpoints, traces, TensorBoard logs. |
| `save_every` | `10` | Checkpoint cadence (batches). Also saves at start. |
| `eval_every` | `10` | Eval cadence (batches). |
| `wandb_project` | `null` | W&B project name (disabled when null). |
| `wandb_name` | `null` | W&B run name. |
| `tensorboard_enabled` | `true` | Enable TensorBoard logging. |
| `tensorboard_log_histograms_every` | `5` | Histogram logging cadence. |
| `tensorboard_log_per_level` | `true` | Log per-level metrics. |
| `base_url` | `null` | Optional custom Tinker base URL. |
| `load_checkpoint_path` | `null` | Start from a specific checkpoint state. |
| `resume_from_batch` | `null` | Resume from a given batch id (otherwise last checkpoint). |
| `dataset_builder.*` | see below | Single-turn dataset settings (also used to seed multi-turn if `multiturn_dataset_builder` is not set). |
| `multiturn_dataset_builder.*` | see below | Optional explicit multi-turn dataset settings. |

## Single-turn dataset_builder (KernelBenchDatasetBuilder)
| Key | Default | What it controls |
| --- | --- | --- |
| `dataset_builder.level` | `1` | KernelBench level. |
| `dataset_builder.start_problem` | `null` | First problem id (inclusive). |
| `dataset_builder.end_problem` | `null` | Last problem id (inclusive). |
| `dataset_builder.backend` | `"triton"` | Kernel backend (`triton` or `cuda`). |
| `dataset_builder.dataset_src` | `"huggingface"` | Source for problems. |
| `dataset_builder.batch_size` | `4` | Problems per batch. |
| `dataset_builder.group_size` | `4` | Rollouts per problem. |
| `dataset_builder.num_epochs` | `1` | Dataset epochs. |
| `dataset_builder.shuffle` | `true` | Shuffle problems each epoch. |
| `dataset_builder.num_correct_trials` | `5` | Correctness trials per eval. |
| `dataset_builder.measure_performance` | `false` | Measure runtime to enable speed rewards. |
| `dataset_builder.reward_format_weight` | `0.1` | Weight for valid response format. |
| `dataset_builder.reward_compile_weight` | `0.2` | Weight for successful compilation. |
| `dataset_builder.reward_correctness_weight` | `1.0` | Weight for correctness. |
| `dataset_builder.reward_speed_weight` | `0.0` | Weight for speedup reward. |
| `dataset_builder.reward_length_weight` | `0.05` | Tie-breaker encouraging shorter code. |
| `dataset_builder.reward_thinking_weight` | `0.1` | Bonus for `<think>…</think>` content. Set to `0.0` for “no-think” Kevin-style runs. |
| `dataset_builder.renderer_name` | `"qwen3"` | Renderer passed to Tinker. |
| `dataset_builder.test_fraction` | `0.1` | Fraction of problems held for test. |
| `dataset_builder.prompt_option` | `"one_shot"` | Prompt style: `zero_shot`, `one_shot`, `few_shot`, or `raicl`. |
| `dataset_builder.rag_index_path` | `null` | Path to RAG index (required when `prompt_option=raicl`). |
| `dataset_builder.raicl_k` | `3` | Retrieved examples per RA-ICL prompt. |
| `dataset_builder.use_modal` | `true` | Evaluate kernels inside Modal sandboxes. |
| `dataset_builder.modal_gpu_type` | `"A100"` | GPU type for Modal evaluation. |
| `dataset_builder.modal_timeout` | `120.0` | Timeout (s) per kernel eval. |

## Multi-turn settings
- Enable with `mode=multi_turn`. If `multiturn_dataset_builder` is not provided, the trainer clones `dataset_builder` values and adds the multi-turn knobs below.
- Thinking tokens in multi-turn: the environment uses the “no-think” prompt when `reward_thinking_weight <= 0`; set it above zero to include `<think>` blocks and give bonuses.

| Key | Default | What it controls |
| --- | --- | --- |
| `multiturn_dataset_builder.level` | `1` | KernelBench level. |
| `multiturn_dataset_builder.start_problem` | `null` | First problem id (inclusive). |
| `multiturn_dataset_builder.end_problem` | `null` | Last problem id (inclusive). |
| `multiturn_dataset_builder.backend` | `"triton"` | Kernel backend. |
| `multiturn_dataset_builder.dataset_src` | `"huggingface"` | Source for problems. |
| `multiturn_dataset_builder.batch_size` | `4` | Problems per batch. |
| `multiturn_dataset_builder.group_size` | `4` | Rollouts per problem. |
| `multiturn_dataset_builder.num_epochs` | `1` | Dataset epochs. |
| `multiturn_dataset_builder.shuffle` | `true` | Shuffle problems each epoch. |
| `multiturn_dataset_builder.max_turns` | `8` | Max refinement turns. |
| `multiturn_dataset_builder.early_stop_on_correct` | `true` | Stop episode early once correctness (and optional speedup) is hit. |
| `multiturn_dataset_builder.speedup_threshold` | `null` | Require this speedup for early stop (e.g., `1.0` to insist on >1x). |
| `multiturn_dataset_builder.num_correct_trials` | `5` | Correctness trials per eval. |
| `multiturn_dataset_builder.measure_performance` | `false` | Measure runtime to unlock speed rewards. |
| `multiturn_dataset_builder.reward_format_weight` | `0.1` | Format reward weight. |
| `multiturn_dataset_builder.reward_compile_weight` | `0.2` | Compile reward weight. |
| `multiturn_dataset_builder.reward_correctness_weight` | `1.0` | Correctness reward weight. |
| `multiturn_dataset_builder.reward_speed_weight` | `0.0` | Speed reward weight. |
| `multiturn_dataset_builder.reward_length_weight` | `0.05` | Length tie-breaker weight. |
| `multiturn_dataset_builder.reward_thinking_weight` | `0.1` | Thinking reward weight; set to `0.0` for Kevin “no-think” prompts. |
| `multiturn_dataset_builder.renderer_name` | `"qwen3"` | Renderer used for prompts. |
| `multiturn_dataset_builder.test_fraction` | `0.1` | Fraction of problems held for test. |
| `multiturn_dataset_builder.prompt_option` | `"raicl"` | Prompt style (RA-ICL recommended). |
| `multiturn_dataset_builder.rag_index_path` | `null` | Path to RAG index (required when `prompt_option=raicl`). |
| `multiturn_dataset_builder.raicl_k` | `3` | Retrieved examples per RA-ICL prompt. |
| `multiturn_dataset_builder.use_modal` | `true` | Evaluate kernels inside Modal sandboxes. |
| `multiturn_dataset_builder.modal_gpu_type` | `"A100"` | GPU type for Modal evaluation. |
| `multiturn_dataset_builder.modal_timeout` | `120.0` | Timeout (s) per kernel eval. |

## Environment variables
- `TINKER_API_KEY` (required): access to the Tinker API.
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`: needed when `use_modal=true`.
- `KERNELBENCH_ROOT`: auto-detected if not set; used for problem assets.
