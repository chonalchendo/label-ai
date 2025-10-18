from pathlib import Path

import polars as pl
import tiktoken


def plan_cost(
    data: pl.DataFrame,
    models: list[str],
    template_path: Path,
    label_column: str,
    context_columns: list[str],
    expected_output_token_length: int,
    models_csv_path: Path,
) -> pl.DataFrame:
    """
    Estimate total dataset labeling cost for each selected LLM model.
    """

    # --- 1. Load model pricing and filter selected models
    model_df = (
        pl.read_csv(models_csv_path)
        .filter(pl.col("slug").is_in(models))
        .select(
            [
                "slug",
                "name",
                "author",
                "provider",
                "context_length",
                "prompt_price",
                "completion_price",
            ]
        )
    )

    if model_df.is_empty():
        raise ValueError(f"No matching models found for slugs: {models}")

    # --- 2. Load prompt template
    template = template_path.read_text()

    # --- 3. Setup tokenizer
    enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    # --- 4. Build prompts and count tokens
    prompt_texts = [
        template.format(
            title=row[context_columns[0]],
            abstract=row[context_columns[1]],
            label_list=row[label_column],
        )
        for row in data.iter_rows(named=True)
    ]

    prompt_token_counts = [count_tokens(text) for text in prompt_texts]
    total_input_tokens = sum(prompt_token_counts)
    n_samples = len(prompt_texts)
    total_output_tokens = n_samples * expected_output_token_length

    # --- 5. Compute total cost for each model
    model_df = model_df.with_columns(
        [
            (
                (pl.lit(total_input_tokens) / 1_000_000) * pl.col("prompt_price")
                + (pl.lit(total_output_tokens) / 1_000_000) * pl.col("completion_price")
            ).alias("total_cost_usd")
        ]
    )

    # --- 6. Add metadata
    model_df = model_df.with_columns(
        [
            pl.lit(total_input_tokens).alias("total_input_tokens"),
            pl.lit(total_output_tokens).alias("total_output_tokens"),
            pl.lit(n_samples).alias("n_samples"),
        ]
    )

    return model_df.select(
        [
            "slug",
            "name",
            "author",
            "provider",
            "context_length",
            "prompt_price",
            "completion_price",
            "total_input_tokens",
            "total_output_tokens",
            "n_samples",
            "total_cost_usd",
        ]
    )
