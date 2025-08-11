import asyncio
from dataclasses import dataclass

import pandas as pd
from openai import AsyncOpenAI
from rich import print

from labelai.cost import calculate_cost


@dataclass
class LabelResult:
    text: str
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


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
    response = await client.responses.create(
        model=model, input=[{"role": "user", "content": prompt}]
    )
    label = response.output_text
    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
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
    model: str,
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

    # Create tasks for all records
    tasks = [label_record(text, labels, model, client) for text in df[text_column]]

    # Await all tasks to complete
    results = await asyncio.gather(*tasks)

    # Add labels to dataframe
    df["label"] = [r.label for r in results]

    # Calculate totals
    total_cost = sum(r.cost for r in results)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)

    # Simple cost summary
    print("\n[green]✓ Complete![/green]")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average per record: ${total_cost / len(df):.6f}")
    print(f"Projected per 1,000: ${(total_cost / len(df) * 1000):.2f}")

    return df
