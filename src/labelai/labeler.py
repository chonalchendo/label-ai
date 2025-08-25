import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from openai import AsyncOpenAI
from rich import print

from labelai.cost import calculate_cost


@dataclass
class LabelResult:
    text: str
    label: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost: float | None


async def majority_vote_label(
    text: str,
    labels: list[str],
    models: Sequence[str],
    client: AsyncOpenAI,
    threshold: float = 0.5,
) -> list[LabelResult]:
    tasks = [
        label_record(text=text, labels=labels, model=model, client=client)
        for model in models
    ]
    results = await asyncio.gather(*tasks)

    counter = Counter(result.label for result in results)

    majority_label = None
    top_label, freq = counter.most_common(1)[0]

    if freq >= threshold:
        majority_label = top_label

    # Add majority vote result to the list
    results.append(
        LabelResult(
            text=text,
            label=majority_label,
            model="majority_vote",
            input_tokens=sum(r.input_tokens for r in results if r.input_tokens),
            output_tokens=sum(r.output_tokens for r in results if r.output_tokens),
            cost=sum(r.cost for r in results if r.cost),
        )
    )

    return results


async def label_record(
    text: str, labels: list[str], model: str, client: AsyncOpenAI
) -> LabelResult:
    """Label a single text record using an LLM.
    Args:
        text (str): The text to classify
        labels (list[str]): List of possible label categories
        model (str): Name of the OpenAI model to use
        client (AsyncOpenAI): Initialized OpenAI async client
    Returns:
        LabelResult: Contains the label, token usage, and cost information
    """
    prompt = f"Classify into {labels}: {text}. Output must only contain the label."
    response = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    label = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    # Rich printing for debugging
    print(f"[green]Label:[/green] {label}")
    print(
        f"[blue]Tokens:[/blue] {input_tokens} in + {output_tokens} out = {total_tokens} total"
    )
    print(f"[yellow]Cost:[/yellow] ${cost:.6f}")
    return LabelResult(
        text=text,
        label=label,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
    )


async def label_dataset(
    df: pd.DataFrame,
    text_column: str,
    labels: list[str],
    models: Sequence[str],
    client: AsyncOpenAI,
) -> pd.DataFrame:
    """Label all records in a dataset and add results as a new column.

    Args:
        df (pd.DataFrame): Dataset containing text to label
        text_column (str): Name of the column containing text to classify
        labels (list[str]): List of possible label categories
        model (str): Name of the OpenAI model to use
        client (AsyncOpenAI): Initialized OpenAI async client

    Returns:
        pd.DataFrame: Original dataframe with added 'label' column
    """
    print(f"[bold]Labeling {len(df)} records...[/bold]\n")

    texts = df[text_column]
    all_results = []

    if len(models) > 1:
        tasks = [majority_vote_label(text, labels, models, client) for text in texts]
        results_lists = await asyncio.gather(*tasks)

        # Flatten results and organize by model
        for results_list in results_lists:
            all_results.extend(results_list[:-1])  # Exclude majority vote for now

        # Add columns for each model
        for model in models:
            model_labels = []
            for i, text in enumerate(texts):
                model_result = results_lists[i]
                model_label = next(
                    (r.label for r in model_result if r.model == model), None
                )
                model_labels.append(model_label)
            df[f"label_{model}"] = model_labels

        # Add majority vote column
        df["label"] = [results_list[-1].label for results_list in results_lists]

        # Include majority vote results in all_results for cost calculation
        for results_list in results_lists:
            all_results.append(results_list[-1])
    else:
        # Create tasks for all records
        tasks = [label_record(text, labels, models[0], client) for text in texts]
        # Await all tasks to complete
        results = await asyncio.gather(*tasks)
        all_results = results
        # Add labels to dataframe
        df["label"] = [r.label for r in results]

    # Calculate totals
    total_cost = sum(r.cost for r in all_results if r.cost)
    total_tokens = sum(
        (r.input_tokens or 0) + (r.output_tokens or 0) for r in all_results
    )

    # Simple cost summary
    print("\n[green]✓ Complete![/green]")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average per record: ${total_cost / len(df):.6f}")
    print(f"Projected per 1,000: ${(total_cost / len(df) * 1000):.2f}")

    return df
