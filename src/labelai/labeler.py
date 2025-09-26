import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from rich import print


@dataclass
class LabelingTask:
    """Represents a single document labeling task."""

    record: dict[str, Any]
    template_path: Path
    context_columns: list[str]
    label_options: list[tuple[str, str]]  # [(code, label), ...]

    def create_prompt(self) -> str:
        """Generate the prompt for this specific record."""
        template = self.template_path.read_text()

        # Extract context values from record
        context = {col: self.record.get(col, "") for col in self.context_columns}

        # Format label options
        context["label_list"] = "\n".join(
            f"ID: {code}, PATH: {label}" for code, label in self.label_options
        )

        return template.format(**context)

    def get_valid_codes(self) -> list[str]:
        """Return list of valid label codes."""
        return [code for code, _ in self.label_options]


class LabelingClient:
    """Handles all LLM interactions for document labeling."""

    def __init__(self, client: AsyncOpenAI, models: list[str]):
        self.client = client
        self.models = models

    async def get_prediction(self, model: str, prompt: str) -> str:
        """Get a single prediction from a model."""
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30,
            )

            # Extract medtop ID from response
            content = response.choices[0].message.content or ""
            matches = re.findall(r"medtop:\d+", content)
            return matches[-1] if matches else "INVALID"

        except Exception as e:
            print(f"Error with {model}: {e}")
            return "ERROR"

    async def get_all_predictions(self, task: LabelingTask) -> dict[str, str]:
        """Get predictions from all models for a single task."""
        prompt = task.create_prompt()

        predictions = await asyncio.gather(
            *[self.get_prediction(model, prompt) for model in self.models]
        )

        return dict(zip(self.models, predictions))


async def label_dataset(
    df: pd.DataFrame,
    client: AsyncOpenAI,
    models: list[str],
    template_path: Path,
    context_columns: list[str],
    label_column: str = "top_labels",
    batch_size: int = 10,
) -> pd.DataFrame:
    """
    Label an entire dataset using multiple models.

    Returns a DataFrame with original data plus model predictions and majority vote.
    """
    labeling_client = LabelingClient(client, models)
    results = []

    # Process in batches for rate limiting
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]

        # Create tasks for this batch
        tasks = [
            LabelingTask(
                record=row.to_dict(),
                template_path=template_path,
                context_columns=context_columns,
                label_options=row[label_column],
            )
            for _, row in batch.iterrows()
        ]

        # Get predictions for all records in batch
        batch_predictions = await asyncio.gather(
            *[labeling_client.get_all_predictions(task) for task in tasks]
        )

        results.extend(batch_predictions)

        print(f"Processed {min(i + batch_size, len(df))}/{len(df)} records")
        await asyncio.sleep(1)  # Rate limiting

    # Combine original data with predictions
    return format_results(df[context_columns], results, models)


def format_results(
    original_df: pd.DataFrame, predictions: list[dict[str, str]], models: list[str]
) -> pd.DataFrame:
    """Format predictions into a clean DataFrame."""
    result_df = original_df.copy()

    # Add individual model predictions
    for model in models:
        result_df[f"{model}_prediction"] = [
            pred.get(model, "ERROR") for pred in predictions
        ]

    # Calculate majority vote
    result_df["majority_vote"] = result_df.apply(
        lambda row: calculate_majority_vote(
            [row[f"{model}_prediction"] for model in models]
        ),
        axis=1,
    )

    # Calculate agreement percentage
    result_df["agreement_pct"] = result_df.apply(
        lambda row: calculate_agreement(
            [row[f"{model}_prediction"] for model in models], row["majority_vote"]
        ),
        axis=1,
    )

    return result_df


def calculate_majority_vote(predictions: list[str]) -> str:
    """Find the most common valid prediction."""
    valid_preds = [p for p in predictions if p not in ["ERROR", "INVALID"]]
    if not valid_preds:
        return "NO_CONSENSUS"

    counts = Counter(valid_preds)
    return counts.most_common(1)[0][0]


def calculate_agreement(predictions: list[str], majority: str) -> float:
    """Calculate percentage of models that agreed with majority."""
    if majority == "NO_CONSENSUS":
        return 0.0
    valid_preds = [p for p in predictions if p not in ["ERROR", "INVALID"]]
    if not valid_preds:
        return 0.0
    return sum(p == majority for p in valid_preds) / len(valid_preds)
