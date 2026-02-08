#!/usr/bin/env python3
"""
CLI entrypoint for multi-turn test-time inference on KernelBench.

This reproduces the basic multi-turn inference setup:
- 16 parallel trajectories per task
- 8 serial refinement steps per trajectory
- temperature = 0.9
- thinking enabled (do not suppress <think>)

It uses the same prompt/evaluation pipeline as training.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any

import chz
import tinker
from tqdm import tqdm

from tinker_cookbook import renderers

from kernel_rl.env import setup_environment
from kernel_rl.envs.kernelbench_client import set_global_retriever
from kernel_rl.envs.multiturn_kernelbench_env import (
    MultiTurnKernelBenchDatasetBuilder,
    MultiTurnKernelBenchEnv,
)
from kernel_rl.training.loop import do_group_rollout_with_envs
from kernel_rl.training.models import get_renderer_name_for_model
from kernel_rl.training.reward import RewardConfig
from kernel_rl.inference.hf_endpoint import HFEndpointClient
from kernel_rl.inference.multiturn_runner import (
    MultiTurnHistoryEntry,
    run_multiturn_steps,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@chz.chz
class MultiTurnInferenceConfig:
    """Configuration for multi-turn test-time inference."""

    # Model/checkpoint configuration
    checkpoint_path: str = ""  # Path to checkpoint or "tinker://..." path
    model_name: str = "Qwen/QwQ-32B"  # For tokenizer/renderer

    # Sampling backend
    sampling_backend: str = "tinker"  # "tinker" or "hf_endpoint"
    hf_endpoint_url: str | None = None
    hf_api_key: str | None = None

    # Evaluation configuration
    level: int = 1
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"

    # Prompting (RA-ICL optional)
    prompt_option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot", "raicl"
    rag_index_path: str | None = None
    raicl_k: int = 3

    # Multi-turn inference configuration
    group_size: int = 16
    max_turns: int = 8
    batch_size: int = 1  # problems per batch

    # Generation configuration
    max_tokens: int = 16384
    temperature: float = 0.9

    # Evaluation settings
    num_correct_trials: int = 5
    measure_performance: bool = True

    # Thinking enabled (do not suppress <think>)
    thinking_weight: float = 0.1

    # Modal evaluation (optional)
    use_modal: bool = True
    modal_gpu_type: str = "A100"
    modal_timeout: float = 60.0

    # Output
    output_path: str = "./multiturn_inference_results.json"

    # Tinker API
    base_url: str | None = None


@dataclass
class TrajectoryResult:
    """Per-trajectory results for a single problem."""
    trajectory_id: int
    best_speedup: float | None
    best_correct: bool
    best_compiled: bool
    history: list[dict[str, Any]]


@dataclass
class ProblemInferenceResult:
    """Aggregated results for a single problem."""
    level: int
    problem_id: int
    trajectories: list[TrajectoryResult]
    best_at_16: float | None
    avg_at_16: float | None


def _trajectory_best_speedup(traj) -> float | None:
    speedups: list[float] = []
    for trans in traj.transitions:
        if trans.metrics and trans.metrics.get("speedup") is not None:
            speedups.append(float(trans.metrics["speedup"]))
    return max(speedups) if speedups else None


def _trajectory_best_correct(traj) -> bool:
    for trans in traj.transitions:
        if trans.metrics and trans.metrics.get("correctness"):
            return True
    return False


def _trajectory_best_compiled(traj) -> bool:
    for trans in traj.transitions:
        if trans.metrics and trans.metrics.get("compiled"):
            return True
    return False


def _trajectory_history(env: MultiTurnKernelBenchEnv) -> list[dict[str, Any]]:
    history = []
    for item in env.state.history:
        history.append({
            "turn": item.get("turn"),
            "summary": item.get("summary"),
            "eval_result": item.get("eval_result"),
            "score": item.get("score"),
            "kernel": item.get("kernel"),
        })
    return history


def _aggregate_best_speedups(trajectories: list[TrajectoryResult]) -> tuple[float | None, float | None]:
    if not trajectories:
        return None, None
    values = [t.best_speedup if t.best_speedup is not None else 0.0 for t in trajectories]
    return max(values), mean(values)


async def run_inference(cfg: MultiTurnInferenceConfig) -> dict[str, Any]:
    """Run multi-turn inference across the requested problems."""
    sampling_client: tinker.SamplingClient | None = None
    hf_client: HFEndpointClient | None = None

    if cfg.sampling_backend == "tinker":
        service_client = tinker.ServiceClient(base_url=cfg.base_url)
        if cfg.checkpoint_path:
            logger.info(f"Loading checkpoint: {cfg.checkpoint_path}")
            sampling_client = service_client.create_sampling_client(cfg.checkpoint_path)
        else:
            logger.info(f"Using base model: {cfg.model_name}")
            sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)
    else:
        if not cfg.hf_endpoint_url:
            raise ValueError("hf_endpoint_url is required when sampling_backend=hf_endpoint")
        hf_client = HFEndpointClient(cfg.hf_endpoint_url, api_key=cfg.hf_api_key)

    # Renderer
    renderer_name = get_renderer_name_for_model(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name)

    # Build dataset (multi-turn) using the same environment pipeline
    dataset_builder = MultiTurnKernelBenchDatasetBuilder(
        level=cfg.level,
        start_problem=cfg.start_problem,
        end_problem=cfg.end_problem,
        backend=cfg.backend,
        dataset_src=cfg.dataset_src,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        num_epochs=1,
        shuffle=False,
        max_turns=cfg.max_turns,
        num_correct_trials=cfg.num_correct_trials,
        measure_performance=cfg.measure_performance,
        reward_thinking_weight=cfg.thinking_weight,
        renderer_name=renderer_name,
        prompt_option=cfg.prompt_option,
        rag_index_path=cfg.rag_index_path,
        raicl_k=cfg.raicl_k,
        use_modal=cfg.use_modal,
        modal_gpu_type=cfg.modal_gpu_type,
        modal_timeout=cfg.modal_timeout,
    )

    # Tokenizer is optional for renderer; provide if available
    tokenizer = None
    if sampling_client is not None:
        try:
            tokenizer = sampling_client.get_tokenizer()
        except Exception:
            tokenizer = None

    dataset, _ = await dataset_builder(tokenizer=tokenizer)
    num_batches = len(dataset)

    if cfg.sampling_backend != "tinker" and cfg.prompt_option == "raicl":
        if not cfg.rag_index_path:
            raise ValueError("rag_index_path is required when prompt_option=raicl")
        from kernel_rl.rag.retriever import KernelRetriever
        retriever = KernelRetriever.load(cfg.rag_index_path)
        set_global_retriever(retriever)

    logger.info(
        "Starting multi-turn inference: %d batches, group_size=%d, max_turns=%d, temp=%.2f",
        num_batches,
        cfg.group_size,
        cfg.max_turns,
        cfg.temperature,
    )

    problem_results: list[ProblemInferenceResult] = []

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        env_group_builders = dataset.get_batch(batch_idx)

        if cfg.sampling_backend == "tinker":
            if sampling_client is None:
                raise RuntimeError("Tinker sampling client is not initialized")
            results = await asyncio.gather(*[
                do_group_rollout_with_envs(
                    sampling_client,
                    builder,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    do_remove_constant_reward_groups=False,
                )
                for builder in env_group_builders
            ], return_exceptions=True)
        else:
            if hf_client is None:
                raise RuntimeError("HF endpoint client is not initialized")
            stop = renderer.get_stop_sequences()
            results = []
            for builder in env_group_builders:
                trajectories: list[TrajectoryResult] = []
                reward_config = RewardConfig(thinking_weight=cfg.thinking_weight)

                async def run_one() -> list[MultiTurnHistoryEntry]:
                    history: list[MultiTurnHistoryEntry] = []
                    async def generate_fn(messages):
                        return await hf_client.chat_completion(
                            messages=messages,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            stop=stop,
                        )
                    return await run_multiturn_steps(
                        builder.problem,
                        history,
                        cfg.max_turns,
                        generate_fn=generate_fn,
                        include_think=cfg.thinking_weight > 0,
                        reward_config=reward_config,
                        num_correct_trials=cfg.num_correct_trials,
                        measure_performance=cfg.measure_performance,
                        use_modal=cfg.use_modal,
                        modal_timeout=cfg.modal_timeout,
                        early_stop_on_correct=True,
                        speedup_threshold=None,
                    )

                histories = await asyncio.gather(*[run_one() for _ in range(cfg.group_size)])
                for idx, hist in enumerate(histories):
                    best_speedup = None
                    best_correct = False
                    best_compiled = False
                    for entry in hist:
                        eval_result = entry.eval_result
                        if eval_result.get("compiled"):
                            best_compiled = True
                        if eval_result.get("correctness"):
                            best_correct = True
                            speed = eval_result.get("speedup")
                            if speed is not None:
                                best_speedup = max(best_speedup or 0.0, float(speed))
                    trajectories.append(TrajectoryResult(
                        trajectory_id=idx,
                        best_speedup=best_speedup,
                        best_correct=best_correct,
                        best_compiled=best_compiled,
                        history=[
                            {
                                "turn": h.turn,
                                "summary": h.summary,
                                "eval_result": h.eval_result,
                                "score": h.score,
                                "kernel": h.kernel,
                            }
                            for h in hist
                        ],
                    ))
                best_at_16, avg_at_16 = _aggregate_best_speedups(trajectories)
                problem_results.append(ProblemInferenceResult(
                    level=builder.problem.level,
                    problem_id=builder.problem.problem_id,
                    trajectories=trajectories,
                    best_at_16=best_at_16,
                    avg_at_16=avg_at_16,
                ))
            continue

        for builder, res in zip(env_group_builders, results):
            if isinstance(res, Exception):
                logger.error("Group rollout failed for problem %s: %s", builder.problem.problem_id, res)
                problem_results.append(ProblemInferenceResult(
                    level=builder.problem.level,
                    problem_id=builder.problem.problem_id,
                    trajectories=[],
                    best_at_16=None,
                    avg_at_16=None,
                ))
                continue

            tg, envs = res
            if tg is None or envs is None:
                problem_results.append(ProblemInferenceResult(
                    level=builder.problem.level,
                    problem_id=builder.problem.problem_id,
                    trajectories=[],
                    best_at_16=None,
                    avg_at_16=None,
                ))
                continue

            trajectories: list[TrajectoryResult] = []
            for idx, (traj, env) in enumerate(zip(tg.trajectories_G, envs)):
                if not isinstance(env, MultiTurnKernelBenchEnv):
                    continue
                trajectories.append(TrajectoryResult(
                    trajectory_id=idx,
                    best_speedup=_trajectory_best_speedup(traj),
                    best_correct=_trajectory_best_correct(traj),
                    best_compiled=_trajectory_best_compiled(traj),
                    history=_trajectory_history(env),
                ))

            best_at_16, avg_at_16 = _aggregate_best_speedups(trajectories)

            problem_results.append(ProblemInferenceResult(
                level=builder.problem.level,
                problem_id=builder.problem.problem_id,
                trajectories=trajectories,
                best_at_16=best_at_16,
                avg_at_16=avg_at_16,
            ))

    # Aggregate metrics across problems
    best_values = [p.best_at_16 for p in problem_results if p.best_at_16 is not None]
    avg_values = [p.avg_at_16 for p in problem_results if p.avg_at_16 is not None]

    metrics = {
        "num_problems": len(problem_results),
        "best_at_16_mean": float(mean(best_values)) if best_values else None,
        "avg_at_16_mean": float(mean(avg_values)) if avg_values else None,
    }

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "results": [asdict(r) for r in problem_results],
    }


def main() -> None:
    setup_environment()

    cfg = chz.entrypoint(MultiTurnInferenceConfig)

    logger.info("Starting multi-turn KernelBench inference")
    logger.info("Checkpoint: %s", cfg.checkpoint_path or "base model")
    logger.info("Level: %s", cfg.level)
    logger.info("Backend: %s", cfg.backend)

    output = asyncio.run(run_inference(cfg))

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    with open(cfg.output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", cfg.output_path)


if __name__ == "__main__":
    main()
