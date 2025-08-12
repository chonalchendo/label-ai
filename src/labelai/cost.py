from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import tiktoken

PRICING = {
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    }  # per 1m tokens
}


@dataclass
class CostPlanResult:
    total_cost_usd: float
    input_cost_usd: float
    output_cost_usd: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    num_rows: int
    avg_input_tokens_per_row: float
    model: str


def estimate_tokens(text: str, model: str) -> int:
    """
    Estimate token count for a given text using tiktoken.
    Falls back to character-based estimation if tiktoken fails.
    """
    try:
        # Use cl100k_base encoding (used by GPT-4 models)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: estimate ~4 characters per token (rough average for English)
        return len(text) // 4


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single API call"""
    if model not in PRICING:
        print(f"[yellow]Warning: No pricing info for {model}[/yellow]")
        return 0.0

    prices = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    return total_cost


def plan_cost(
    df: pd.DataFrame,
    text_column: str,
    model: str,
    labels: Sequence[str] | None = None,
    system_prompt: str = "",
    expected_output_tokens_per_row: int | None = None,
    include_context: bool = True,
) -> CostPlanResult:
    """Plan the cost of labelling a dataset."""
    # Calculate expected output tokens based on labels if provided
    if expected_output_tokens_per_row is None:
        if labels:
            # Calculate average token count across all labels
            label_tokens = [estimate_tokens(label, model) for label in labels]
            avg_label_tokens = sum(label_tokens) / len(label_tokens)
            # Add buffer for potential explanations or formatting (e.g., "Label: positive")
            # Adjust this multiplier based on your expected response format
            expected_output_tokens_per_row = int(
                avg_label_tokens * 2
            )  # 2x for some context
            print(
                f"Estimated output tokens per row based on labels: {expected_output_tokens_per_row}"
            )
        else:
            # Default fallback if no labels provided
            expected_output_tokens_per_row = 50
            print(
                f"Using default output tokens per row: {expected_output_tokens_per_row}"
            )

    # Calculate system prompt tokens (if any)
    system_tokens = estimate_tokens(system_prompt, model) if system_prompt else 0

    # Calculate the total input tokens from dataset
    total_input_tokens = 0
    for text in df[text_column].dropna():
        # Estimate tokens for each text entry
        text_tokens = estimate_tokens(str(text), model)

        # Add system prompt tokens if including context for each row
        if include_context:
            total_input_tokens += text_tokens + system_tokens
        else:
            total_input_tokens += text_tokens

    # Add system tokens once if batch processing
    if not include_context and system_prompt:
        total_input_tokens += system_tokens

    # Calculate the total output tokens from dataset
    # This is an estimate - adjust based on your expected response length
    num_rows = len(df[text_column].dropna())
    total_output_tokens = num_rows * expected_output_tokens_per_row

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (total_output_tokens / 1_000_000) * PRICING[model]["output"]
    total_cost = input_cost + output_cost

    # Return detailed breakdown
    return CostPlanResult(
        total_cost_usd=round(total_cost, 4),
        input_cost_usd=round(input_cost, 4),
        output_cost_usd=round(output_cost, 4),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_input_tokens + total_output_tokens,
        num_rows=num_rows,
        avg_input_tokens_per_row=total_input_tokens / num_rows if num_rows > 0 else 0,
        model=model,
    )
