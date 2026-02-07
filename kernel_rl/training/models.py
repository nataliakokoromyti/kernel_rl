"""
Tinker model and client configuration for KernelBench RL training.

This module provides helpers for:
- Creating Tinker ServiceClient and TrainingClient
- Configuring LoRA for code models
- Model selection from the Tinker lineup
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

import tinker
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


# Recommended models for kernel generation
# These are Qwen coder models which are well-suited for code tasks
RECOMMENDED_MODELS = {
    # Smaller models for development/testing
    "qwen-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",

    # Larger models for production
    "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",

    # Alternative code models
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
    "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",

    # General models that work for code
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",

    # MoE models
    "qwen-moe": "Qwen/Qwen3-235B-A22B",
}

# Default model for training
DEFAULT_MODEL = "qwen-coder-7b"


@dataclass
class ModelConfig:
    """Configuration for model training."""

    # Model selection
    model_name: str = RECOMMENDED_MODELS[DEFAULT_MODEL]

    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # Training configuration
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    temperature: float = 1.0

    # Tinker API configuration
    base_url: str | None = None

    def get_model_name(self) -> str:
        """Get the full model name, resolving shortcuts."""
        if self.model_name in RECOMMENDED_MODELS:
            return RECOMMENDED_MODELS[self.model_name]
        return self.model_name


def create_service_client(
    base_url: str | None = None,
) -> tinker.ServiceClient:
    """
    Create a Tinker ServiceClient.

    Args:
        base_url: Optional Tinker API base URL

    Returns:
        Configured ServiceClient
    """
    return tinker.ServiceClient(base_url=base_url)


async def create_training_client(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    checkpoint_path: str | None = None,
) -> tinker.TrainingClient:
    """
    Create a Tinker TrainingClient for RL training.

    Args:
        service_client: Tinker service client
        model_config: Model configuration
        checkpoint_path: Optional path to resume from checkpoint

    Returns:
        Configured TrainingClient with LoRA
    """
    model_name = model_config.get_model_name()

    if checkpoint_path:
        # Resume from checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        training_client = await service_client.create_training_client_from_state_async(
            checkpoint_path
        )
    else:
        # Create new LoRA training client
        logger.info(f"Creating LoRA training client for {model_name}")
        training_client = await service_client.create_lora_training_client_async(
            model_name,
            rank=model_config.lora_rank,
        )

    return training_client


def get_tokenizer_for_model(model_name: str) -> Tokenizer:
    """
    Get the tokenizer for a model.

    Args:
        model_name: Full model name (HuggingFace path)

    Returns:
        Tokenizer instance
    """
    # Resolve shortcut names
    if model_name in RECOMMENDED_MODELS:
        model_name = RECOMMENDED_MODELS[model_name]

    return get_tokenizer(model_name)


def get_renderer_name_for_model(model_name: str) -> str:
    """
    Get the appropriate renderer name for a model.

    Args:
        model_name: Full model name

    Returns:
        Renderer name (e.g., "qwen3", "llama3")
    """
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        return "qwen3"
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return "llama3"
    elif "codellama" in model_lower:
        return "llama3"  # CodeLlama uses Llama format
    else:
        # Default to role_colon for unknown models
        return "role_colon"


def estimate_training_cost(
    num_problems: int,
    num_epochs: int,
    batch_size: int,
    group_size: int,
    max_tokens: int,
    model_name: str,
) -> dict[str, Any]:
    """
    Estimate training cost and time.

    This is a rough estimate based on typical training characteristics.

    Args:
        num_problems: Number of problems in dataset
        num_epochs: Number of training epochs
        batch_size: Problems per batch
        group_size: Rollouts per problem
        max_tokens: Maximum tokens per generation
        model_name: Model name

    Returns:
        Dictionary with estimates
    """
    total_rollouts = num_problems * num_epochs * group_size
    total_batches = (num_problems * num_epochs + batch_size - 1) // batch_size

    # Rough token estimates
    prompt_tokens = 1500  # Average prompt length for KernelBench
    completion_tokens = max_tokens // 2  # Assume average half of max
    tokens_per_rollout = prompt_tokens + completion_tokens

    total_tokens = total_rollouts * tokens_per_rollout

    return {
        "total_rollouts": total_rollouts,
        "total_batches": total_batches,
        "estimated_tokens": total_tokens,
        "estimated_tokens_millions": total_tokens / 1_000_000,
        "model": model_name,
    }


# Adam optimizer parameters (matching Tinker cookbook defaults)
def get_adam_params(learning_rate: float) -> tinker.AdamParams:
    """Get Adam optimizer parameters."""
    return tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip_norm=0.05,
    )
