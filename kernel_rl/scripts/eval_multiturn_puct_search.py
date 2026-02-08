#!/usr/bin/env python3
"""
CLI entrypoint for multi-turn test-time PUCT search on KernelBench.

Implements PUCT-based state reuse:
- Seed with initial rollouts from empty history
- Maintain a buffer of all states
- Select which state to expand via PUCT scoring

Prompting, evaluation, and refinement flow are identical to the
multi-turn training environment.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@chz.chz
class PUCTSearchConfig:
    """Configuration for multi-turn PUCT search inference."""

    # Model/checkpoint configuration
    checkpoint_path: str = ""  # Path to checkpoint or "tinker://..." path
    model_name: str = "Qwen/QwQ-32B"  # For tokenizer/renderer

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

    # PUCT configuration
    total_rollouts: int = 128
    init_rollouts: int = 16
    steps_per_rollout: int = 4
    exploration_coeff: float = 1.0

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
    output_path: str = "./multiturn_puct_search_results.json"

    # Tinker API
    base_url: str | None = None


@dataclass
class BufferState:
    history: list[dict[str, Any]]
    best_speedup: float
    visit_count: int
    best_child_speedup: float


@dataclass
class TrajectoryResult:
    trajectory_id: int
    best_speedup: float
    best_correct: bool
    best_compiled: bool
    history: list[dict[str, Any]]


@dataclass
class ProblemPUCTResult:
    level: int
    problem_id: int
    trajectories: list[TrajectoryResult]
    best_speedup: float
    mean_speedup: float
    best_kernel: str | None


class PUCTBuffer:
    def __init__(self, exploration_coeff: float = 1.0) -> None:
        self.c = exploration_coeff
        self.states: list[BufferState] = []
        self.total_expansions = 0

    def add_state(self, history: list[dict[str, Any]], speedup: float) -> None:
        self.states.append(BufferState(
            history=history,
            best_speedup=speedup,
            visit_count=0,
            best_child_speedup=speedup,
        ))

    def select_state(self) -> int:
        if not self.states:
            raise RuntimeError("PUCT buffer is empty")

        sorted_indices = sorted(
            range(len(self.states)),
            key=lambda i: self.states[i].best_speedup,
            reverse=True,
        )

        rank_prior: dict[int, float] = {}
        for rank, idx in enumerate(sorted_indices):
            rank_prior[idx] = 1.0 / (rank + 1)

        total_prior = sum(rank_prior.values())
        for idx in rank_prior:
            rank_prior[idx] /= total_prior

        best_score = -float("inf")
        best_idx = 0
        T = self.total_expansions

        for idx, state in enumerate(self.states):
            Q = state.best_child_speedup
            P = rank_prior[idx]
            n = state.visit_count
            score = Q + self.c * P * math.sqrt(1 + T) / (1 + n)
            if score > best_score:
                best_score = score
                best_idx = idx

        self.states[best_idx].visit_count += 1
        self.total_expansions += 1
        return best_idx

    def update_child_speedup(self, parent_idx: int, child_speedup: float) -> None:
        state = self.states[parent_idx]
        if child_speedup > state.best_child_speedup:
            state.best_child_speedup = child_speedup


def _strip_history(env_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _clone_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(history)


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


async def run_puct_search_for_problem(
    problem: KernelBenchProblem,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    cfg: PUCTSearchConfig,
) -> ProblemPUCTResult:
    if cfg.init_rollouts > cfg.total_rollouts:
        raise ValueError("init_rollouts must be <= total_rollouts")

    policy = TinkerTokenCompleter(
        sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    buffer = PUCTBuffer(exploration_coeff=cfg.exploration_coeff)
    reward_config = RewardConfig(thinking_weight=cfg.thinking_weight)

    async def run_one(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_turns = len(history) + cfg.steps_per_rollout
        env = MultiTurnKernelBenchEnv(
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
        )
        env = await _run_env_steps(env, history, policy)
        return _strip_history(env.state.history)

    # Seed buffer with initial exploration
    seed_results = await asyncio.gather(*[
        run_one([])
        for _ in range(cfg.init_rollouts)
    ], return_exceptions=True)

    for res in seed_results:
        if isinstance(res, Exception):
            logger.error("Seed rollout failed: %s", res)
            buffer.add_state([], 0.0)
            continue
        best_speedup, _, _, _ = _best_from_history(res)
        buffer.add_state(res, best_speedup)

    remaining = cfg.total_rollouts - cfg.init_rollouts
    for _ in range(remaining):
        parent_idx = buffer.select_state()
        parent_history = _clone_history(buffer.states[parent_idx].history)

        try:
            child_history = await run_one(parent_history)
        except Exception as exc:
            logger.error("Expansion failed: %s", exc)
            child_history = []

        best_speedup, _, _, _ = _best_from_history(child_history)
        buffer.add_state(child_history, best_speedup)
        buffer.update_child_speedup(parent_idx, best_speedup)

    trajectories: list[TrajectoryResult] = []
    best_kernel = None
    best_speedup_overall = 0.0

    for idx, state in enumerate(buffer.states):
        best_speedup, kernel, best_correct, best_compiled = _best_from_history(state.history)
        if best_speedup > best_speedup_overall:
            best_speedup_overall = best_speedup
            best_kernel = kernel
        trajectories.append(TrajectoryResult(
            trajectory_id=idx,
            best_speedup=best_speedup,
            best_correct=best_correct,
            best_compiled=best_compiled,
            history=state.history,
        ))

    mean_speedup = mean([t.best_speedup for t in trajectories]) if trajectories else 0.0

    return ProblemPUCTResult(
        level=problem.level,
        problem_id=problem.problem_id,
        trajectories=trajectories,
        best_speedup=best_speedup_overall,
        mean_speedup=mean_speedup,
        best_kernel=best_kernel,
    )


async def run_inference(cfg: PUCTSearchConfig) -> dict[str, Any]:
    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    if cfg.checkpoint_path:
        logger.info("Loading checkpoint: %s", cfg.checkpoint_path)
        sampling_client = service_client.create_sampling_client(cfg.checkpoint_path)
    else:
        logger.info("Using base model: %s", cfg.model_name)
        sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

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

    logger.info("Running PUCT search on %d problems", len(problems))

    results: list[ProblemPUCTResult] = []
    for problem in tqdm(problems, desc="PUCT search"):
        try:
            res = await run_puct_search_for_problem(
                problem, sampling_client, renderer, cfg
            )
            results.append(res)
        except Exception as exc:
            logger.error("Problem %s failed: %s", problem.problem_id, exc)
            results.append(ProblemPUCTResult(
                level=problem.level,
                problem_id=problem.problem_id,
                trajectories=[],
                best_speedup=0.0,
                mean_speedup=0.0,
                best_kernel=None,
            ))

    metrics = {
        "num_problems": len(results),
        "best_speedup_mean": float(mean([r.best_speedup for r in results])) if results else None,
        "mean_speedup_mean": float(mean([r.mean_speedup for r in results])) if results else None,
    }

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    setup_environment()

    cfg = chz.entrypoint(PUCTSearchConfig)

    logger.info("Starting multi-turn PUCT search inference")
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
