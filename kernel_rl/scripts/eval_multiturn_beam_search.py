#!/usr/bin/env python3
"""
CLI entrypoint for multi-turn test-time beam search on KernelBench.

Implements Kevin-style beam search:
- Round 1: num_beams trajectories, steps_per_round refinements
- Subsequent rounds: keep top beam_width trajectories, clone each to
  restore num_beams, and run steps_per_round more refinements.

Prompting, evaluation, and refinement flow are identical to the
multi-turn training environment.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any

import chz
import tinker
from tqdm import tqdm

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook import renderers

from kernel_rl.env import setup_environment
from kernel_rl.envs.kernelbench_client import (
    KernelBenchProblem,
    get_problem_ids,
    set_global_retriever,
)
from kernel_rl.envs.multiturn_kernelbench_env import MultiTurnKernelBenchEnv
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
class BeamSearchConfig:
    """Configuration for multi-turn beam search inference."""

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

    # Beam search configuration
    num_beams: int = 16
    beam_width: int = 4
    steps_per_round: int = 4
    num_rounds: int = 2

    # Generation configuration
    max_tokens: int = 16384
    temperature: float = 0.9

    # Evaluation settings
    num_correct_trials: int = 5
    measure_performance: bool = True

    # Thinking enabled (do not suppress <think>)
    thinking_weight: float = 0.1

    # Early stop settings
    early_stop_on_correct: bool = True
    speedup_threshold: float | None = None

    # Modal evaluation (optional)
    use_modal: bool = True
    modal_gpu_type: str = "A100"
    modal_timeout: float = 60.0

    # Output
    output_path: str = "./multiturn_beam_search_results.json"

    # Tinker API
    base_url: str | None = None


@dataclass
class TrajectoryResult:
    """Per-trajectory results for a single problem."""
    trajectory_id: int
    best_speedup: float
    best_correct: bool
    best_compiled: bool
    history: list[dict[str, Any]]


@dataclass
class ProblemBeamResult:
    """Aggregated results for a single problem."""
    level: int
    problem_id: int
    trajectories: list[TrajectoryResult]
    best_at_16: float
    avg_at_16: float
    best_kernel: str | None


def _strip_history(env_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep summary only (discard CoT) while preserving eval results."""
    cleaned = []
    for item in env_history:
        eval_result = item.get("eval_result")
        speedup = 0.0
        if eval_result and eval_result.get("correctness") and eval_result.get("speedup") is not None:
            speedup = float(eval_result.get("speedup") or 0.0)
        cleaned.append({
            "turn": item.get("turn"),
            "kernel": item.get("kernel"),
            "summary": item.get("summary"),
            "eval_result": eval_result,
            "score": float(item.get("score", 0.0)),
            "speedup": speedup,
        })
    return cleaned


def _best_from_history(history: list[dict[str, Any]]) -> tuple[float, str | None, bool, bool]:
    best_speedup = 0.0
    best_kernel = None
    best_correct = False
    best_compiled = False

    for item in history:
        eval_result = item.get("eval_result") or {}
        compiled = bool(eval_result.get("compiled"))
        correct = bool(eval_result.get("correctness"))
        if compiled:
            best_compiled = True
        if correct:
            best_correct = True
            speedup = float(eval_result.get("speedup") or 0.0)
            if speedup > best_speedup:
                best_speedup = speedup
                best_kernel = item.get("kernel")

    return best_speedup, best_kernel, best_correct, best_compiled


def _clone_histories(histories: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return [copy.deepcopy(h) for h in histories]


async def _run_env_steps(
    env: MultiTurnKernelBenchEnv,
    history: list[dict[str, Any]],
    policy: TinkerTokenCompleter,
) -> MultiTurnKernelBenchEnv:
    if history:
        obs, stop = await env.initial_observation_from_history(history)
    else:
        obs, stop = await env.initial_observation()

    while True:
        action = await policy(obs, stop)
        step = await env.step(action)
        if step.episode_done:
            break
        obs = step.next_observation
        stop = step.next_stop_condition

    return env


async def run_beam_search_for_problem(
    problem: KernelBenchProblem,
    sampling_client: tinker.SamplingClient | None,
    hf_client: HFEndpointClient | None,
    renderer: renderers.Renderer,
    cfg: BeamSearchConfig,
) -> ProblemBeamResult:
    if cfg.num_beams % cfg.beam_width != 0:
        raise ValueError("num_beams must be divisible by beam_width")

    policy = None
    if cfg.sampling_backend == "tinker":
        if sampling_client is None:
            raise RuntimeError("Tinker sampling client is not initialized")
        policy = TinkerTokenCompleter(
            sampling_client,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

    histories: list[list[dict[str, Any]]] = [[] for _ in range(cfg.num_beams)]

    for round_idx in range(cfg.num_rounds):
        next_histories: list[list[dict[str, Any]]] = []
        if cfg.sampling_backend == "tinker":
            envs: list[MultiTurnKernelBenchEnv] = []
            for history in histories:
                total_turns = len(history) + cfg.steps_per_round
                reward_config = RewardConfig(thinking_weight=cfg.thinking_weight)
                envs.append(MultiTurnKernelBenchEnv(
                    problem=problem,
                    renderer=renderer,
                    max_turns=total_turns,
                    reward_config=reward_config,
                    num_correct_trials=cfg.num_correct_trials,
                    measure_performance=cfg.measure_performance,
                    early_stop_on_correct=cfg.early_stop_on_correct,
                    speedup_threshold=cfg.speedup_threshold,
                    use_modal=cfg.use_modal,
                    modal_timeout=cfg.modal_timeout,
                ))

            results = await asyncio.gather(*[
                _run_env_steps(env, history, policy)
                for env, history in zip(envs, histories)
            ], return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    logger.error("Trajectory failed: %s", res)
                    next_histories.append([])
                    continue
                cleaned = _strip_history(res.state.history)
                next_histories.append(cleaned)
        else:
            if hf_client is None:
                raise RuntimeError("HF endpoint client is not initialized")
            stop = renderer.get_stop_sequences()
            reward_config = RewardConfig(thinking_weight=cfg.thinking_weight)

            async def run_one(history: list[dict[str, Any]]) -> list[MultiTurnHistoryEntry]:
                history_entries = [
                    MultiTurnHistoryEntry(
                        turn=h.get("turn", 0),
                        kernel=h.get("kernel"),
                        summary=h.get("summary"),
                        eval_result=h.get("eval_result") or {},
                        score=float(h.get("score", 0.0)),
                    )
                    for h in history
                ]

                async def generate_fn(messages):
                    return await hf_client.chat_completion(
                        messages=messages,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        stop=stop,
                    )

                return await run_multiturn_steps(
                    problem,
                    history_entries,
                    cfg.steps_per_round,
                    generate_fn=generate_fn,
                    include_think=cfg.thinking_weight > 0,
                    reward_config=reward_config,
                    num_correct_trials=cfg.num_correct_trials,
                    measure_performance=cfg.measure_performance,
                    use_modal=cfg.use_modal,
                    modal_timeout=cfg.modal_timeout,
                    early_stop_on_correct=cfg.early_stop_on_correct,
                    speedup_threshold=cfg.speedup_threshold,
                )

            results = await asyncio.gather(*[
                run_one(history) for history in histories
            ], return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    logger.error("Trajectory failed: %s", res)
                    next_histories.append([])
                    continue
                next_histories.append([
                    {
                        "turn": h.turn,
                        "kernel": h.kernel,
                        "summary": h.summary,
                        "eval_result": h.eval_result,
                        "score": h.score,
                    }
                    for h in res
                ])

        histories = next_histories

        if round_idx < cfg.num_rounds - 1:
            scored = []
            for history in histories:
                best_speedup, _, _, _ = _best_from_history(history)
                scored.append((best_speedup, history))

            scored.sort(key=lambda x: x[0], reverse=True)
            survivors = [h for _, h in scored[: cfg.beam_width]]
            clones_per = cfg.num_beams // cfg.beam_width
            histories = []
            for survivor in survivors:
                histories.extend(_clone_histories([survivor] * clones_per))

    trajectories: list[TrajectoryResult] = []
    best_kernel = None
    best_speedup_overall = 0.0

    for idx, history in enumerate(histories):
        best_speedup, kernel, best_correct, best_compiled = _best_from_history(history)
        if best_speedup > best_speedup_overall:
            best_speedup_overall = best_speedup
            best_kernel = kernel
        trajectories.append(TrajectoryResult(
            trajectory_id=idx,
            best_speedup=best_speedup,
            best_correct=best_correct,
            best_compiled=best_compiled,
            history=history,
        ))

    best_at_16 = max(t.best_speedup for t in trajectories) if trajectories else 0.0
    avg_at_16 = mean(t.best_speedup for t in trajectories) if trajectories else 0.0

    return ProblemBeamResult(
        level=problem.level,
        problem_id=problem.problem_id,
        trajectories=trajectories,
        best_at_16=best_at_16,
        avg_at_16=avg_at_16,
        best_kernel=best_kernel,
    )


async def run_inference(cfg: BeamSearchConfig) -> dict[str, Any]:
    sampling_client: tinker.SamplingClient | None = None
    hf_client: HFEndpointClient | None = None

    if cfg.sampling_backend == "tinker":
        service_client = tinker.ServiceClient(base_url=cfg.base_url)
        if cfg.checkpoint_path:
            logger.info("Loading checkpoint: %s", cfg.checkpoint_path)
            sampling_client = service_client.create_sampling_client(cfg.checkpoint_path)
        else:
            logger.info("Using base model: %s", cfg.model_name)
            sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)
    else:
        if not cfg.hf_endpoint_url:
            raise ValueError("hf_endpoint_url is required when sampling_backend=hf_endpoint")
        hf_client = HFEndpointClient(cfg.hf_endpoint_url, api_key=cfg.hf_api_key)

    renderer_name = get_renderer_name_for_model(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name)

    if cfg.prompt_option == "raicl":
        if not cfg.rag_index_path:
            raise ValueError("rag_index_path is required when prompt_option=raicl")
        from kernel_rl.rag.retriever import KernelRetriever
        retriever = KernelRetriever.load(cfg.rag_index_path)
        set_global_retriever(retriever)

    problem_ids = get_problem_ids(
        cfg.level,
        start=cfg.start_problem,
        end=cfg.end_problem,
        dataset_src=cfg.dataset_src,
    )

    problems = [
        KernelBenchProblem(
            level=cfg.level,
            problem_id=pid,
            backend=cfg.backend,
            dataset_src=cfg.dataset_src,
            prompt_option=cfg.prompt_option,
            raicl_k=cfg.raicl_k,
        )
        for pid in problem_ids
    ]

    logger.info("Running beam search on %d problems", len(problems))

    results: list[ProblemBeamResult] = []
    for problem in tqdm(problems, desc="Beam search"):
        try:
            res = await run_beam_search_for_problem(
                problem, sampling_client, hf_client, renderer, cfg
            )
            results.append(res)
        except Exception as exc:
            logger.error("Problem %s failed: %s", problem.problem_id, exc)
            results.append(ProblemBeamResult(
                level=problem.level,
                problem_id=problem.problem_id,
                trajectories=[],
                best_at_16=0.0,
                avg_at_16=0.0,
                best_kernel=None,
            ))

    metrics = {
        "num_problems": len(results),
        "best_at_16_mean": float(mean([r.best_at_16 for r in results])) if results else None,
        "avg_at_16_mean": float(mean([r.avg_at_16 for r in results])) if results else None,
    }

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    setup_environment()

    cfg = chz.entrypoint(BeamSearchConfig)

    logger.info("Starting multi-turn beam search inference")
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
