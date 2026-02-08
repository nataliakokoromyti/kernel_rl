"""
Shared multi-turn inference runner for non-Tinker backends.

Reuses the same prompts and evaluation logic as MultiTurnKernelBenchEnv,
but drives generation via an external chat-completions API (e.g., HF endpoints).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kernel_rl.envs.kernelbench_client import (
    KernelBenchProblem,
    evaluate_kernel,
    evaluate_kernel_async,
    parse_structured_response,
)
from kernel_rl.envs.multiturn_kernelbench_env import (
    MULTITURN_SYSTEM_PROMPT_NO_THINK,
    MULTITURN_SYSTEM_PROMPT_WITH_THINK,
    REFINEMENT_TEMPLATE,
    ERROR_SECTION_TEMPLATE,
    _categorize_error,
    _extract_key_error,
    _fallback_summary,
    _get_error_guidance,
    _truncate_kernel,
    _truncate_summary,
)
from kernel_rl.training.reward import RewardConfig, compute_reward


@dataclass
class MultiTurnHistoryEntry:
    turn: int
    kernel: str | None
    summary: str | None
    eval_result: dict[str, Any]
    score: float


def _build_system_prompt(backend: str, include_think: bool) -> str:
    template = MULTITURN_SYSTEM_PROMPT_WITH_THINK if include_think else MULTITURN_SYSTEM_PROMPT_NO_THINK
    return template.format(backend=backend.upper())


def _build_messages(
    problem: KernelBenchProblem,
    history: list[MultiTurnHistoryEntry],
    include_think: bool,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_prompt = _build_system_prompt(problem.backend, include_think)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_parts = [problem.prompt]

    if history:
        last = history[-1]
        eval_result = last.eval_result

        error_category = _categorize_error(eval_result)
        error_category_display = {
            "format_error": "FORMAT ERROR - Invalid code structure",
            "compilation_error": "COMPILATION ERROR - Build failed",
            "runtime_error": "RUNTIME ERROR - Crashed during execution",
            "correctness_error": "CORRECTNESS ERROR - Wrong output",
            "performance_warning": "CORRECT (but slower than baseline)",
            "success": "SUCCESS",
        }.get(error_category, error_category.upper())

        speedup_line = ""
        if eval_result.get("speedup") is not None:
            speedup_line = f"- **Speedup**: {eval_result['speedup']:.2f}x"

        error_section = ""
        if eval_result.get("error_message") and error_category != "success":
            error_text = _extract_key_error(eval_result.get("error_message"))
            if error_text:
                error_section = ERROR_SECTION_TEMPLATE.format(error_text=error_text)

        guidance = _get_error_guidance(error_category, problem.backend)
        if not guidance:
            guidance = "Fix the issues in the previous attempt and try again."

        previous_summary = last.summary or _fallback_summary(eval_result, error_category)
        previous_summary = _truncate_summary(previous_summary)

        refinement_text = REFINEMENT_TEMPLATE.format(
            turn=last.turn,
            previous_summary=previous_summary,
            previous_kernel=_truncate_kernel(last.kernel or ""),
            error_category=error_category_display,
            compiled="Yes" if eval_result.get("compiled") else "No",
            tests_passed=eval_result.get("tests_passed", 0),
            tests_total=eval_result.get("tests_total", 0),
            speedup_line=speedup_line,
            error_section=error_section,
            guidance=guidance,
        )

        if include_think:
            refinement_text += (
                "\nRemember: respond using <think>...</think>, then <KERNEL>...</KERNEL>, then <SUMMARY>...</SUMMARY>."
            )
        else:
            refinement_text += "\nRemember: respond using <KERNEL>...</KERNEL> followed by <SUMMARY>...</SUMMARY>."

        user_parts.append(refinement_text)

    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


async def run_multiturn_steps(
    problem: KernelBenchProblem,
    history: list[MultiTurnHistoryEntry],
    steps: int,
    *,
    generate_fn,
    include_think: bool,
    reward_config: RewardConfig,
    num_correct_trials: int,
    measure_performance: bool,
    use_modal: bool,
    modal_timeout: float,
    early_stop_on_correct: bool,
    speedup_threshold: float | None,
) -> list[MultiTurnHistoryEntry]:
    """
    Run a fixed number of refinement steps, continuing from existing history.

    generate_fn(messages) -> response text (string)
    """
    for step_idx in range(steps):
        messages = _build_messages(problem, history, include_think)
        response_text = await generate_fn(messages)

        parsed = parse_structured_response(response_text)
        kernel_code = parsed.kernel or ""
        summary = _truncate_summary(parsed.thought_summary) if parsed.thought_summary else None

        if use_modal:
            eval_result = await evaluate_kernel_async(
                level=problem.level,
                problem_id=problem.problem_id,
                backend=problem.backend,
                kernel_code=kernel_code,
                dataset_src=problem.dataset_src,
                num_correct_trials=num_correct_trials,
                measure_performance=measure_performance,
                timeout=modal_timeout,
            )
        else:
            eval_result = evaluate_kernel(
                level=problem.level,
                problem_id=problem.problem_id,
                backend=problem.backend,
                kernel_code=kernel_code,
                dataset_src=problem.dataset_src,
                num_correct_trials=num_correct_trials,
                measure_performance=measure_performance,
            )

        step_score = compute_reward(
            eval_result,
            reward_config,
            thought_length=len(parsed.thought or ""),
        )

        history.append(MultiTurnHistoryEntry(
            turn=len(history),
            kernel=kernel_code,
            summary=summary,
            eval_result=eval_result,
            score=step_score,
        ))

        if early_stop_on_correct and eval_result.get("correctness"):
            meets_speedup = (
                speedup_threshold is None or
                (eval_result.get("speedup") or 0) >= speedup_threshold
            )
            if meets_speedup:
                break

    return history
