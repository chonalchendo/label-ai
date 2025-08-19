from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd
import tiktoken
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

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
    prompt_template: Callable | None = None,
    additional_columns: Sequence[str] | None = None,
    expected_output_tokens_per_row: int | None = None,
    include_context: bool = True,
    show_progress: bool = True,
) -> CostPlanResult:
    """Plan the cost of labelling a dataset."""
    console = Console()

    # Calculate expected output tokens based on labels if provided
    if expected_output_tokens_per_row is None:
        if labels:
            # For ID-based labels like "medtop:20000842", estimate tokens
            label_tokens = [estimate_tokens(str(label), model) for label in labels]
            avg_label_tokens = sum(label_tokens) / len(label_tokens)
            # Add small buffer since output is just the ID
            expected_output_tokens_per_row = int(
                avg_label_tokens * 1.5
            )  # 1.5x for minimal context
            if show_progress:
                console.print(
                    f"[cyan]Estimated output tokens per row based on labels: {expected_output_tokens_per_row}[/cyan]"
                )
        else:
            # Default fallback if no labels provided
            expected_output_tokens_per_row = 50
            if show_progress:
                console.print(
                    f"[yellow]Using default output tokens per row: {expected_output_tokens_per_row}[/yellow]"
                )

    # Calculate system prompt tokens (if any)
    system_tokens = estimate_tokens(system_prompt, model) if system_prompt else 0

    # Prepare columns to use
    columns_to_use = [text_column]
    if additional_columns:
        columns_to_use.extend(additional_columns)

    # Get pricing info
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])

    # Calculate the total input tokens from dataset
    total_input_tokens = 0
    total_output_tokens = 0
    sample_prompt = None  # Store a sample for debugging

    # Filter out rows with NaN in the text column
    valid_df = df.dropna(subset=[text_column])
    num_rows = len(valid_df)

    if show_progress:
        # Create a layout for the progress display
        def create_display_table(input_tokens, output_tokens, rows_processed):
            """Create a rich table showing current stats"""
            input_cost = (input_tokens / 1_000_000) * prices["input"]
            output_cost = (output_tokens / 1_000_000) * prices["output"]
            total_cost = input_cost + output_cost

            table = Table(
                title=f"Cost Estimation Progress - {model}",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", justify="right", style="green")

            table.add_row("Rows Processed", f"{rows_processed:,} / {num_rows:,}")
            table.add_row("Input Tokens", f"{input_tokens:,}")
            table.add_row("Output Tokens (Est.)", f"{output_tokens:,}")
            table.add_row("Total Tokens", f"{input_tokens + output_tokens:,}")
            table.add_row("", "")  # Spacer
            table.add_row("Input Cost", f"${input_cost:.4f}")
            table.add_row("Output Cost (Est.)", f"${output_cost:.4f}")
            table.add_row("[bold]Total Cost (Est.)", f"[bold]${total_cost:.4f}")

            if rows_processed > 0:
                avg_tokens_per_row = input_tokens / rows_processed
                cost_per_row = total_cost / rows_processed
                table.add_row("", "")  # Spacer
                table.add_row("Avg Input Tokens/Row", f"{avg_tokens_per_row:.1f}")
                table.add_row("Avg Cost/Row", f"${cost_per_row:.6f}")

                # Project final cost
                if rows_processed < num_rows:
                    projected_cost = cost_per_row * num_rows
                    table.add_row(
                        "[yellow]Projected Total", f"[yellow]${projected_cost:.4f}"
                    )

            return Panel(table, expand=False)

        # Set up progress bar
        with Progress(
            TextColumn("[bold blue]Processing rows", justify="right"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("[cyan]Estimating costs...", total=num_rows)

            with Live(
                create_display_table(0, 0, 0), console=console, refresh_per_second=4
            ) as live:
                rows_processed = 0

                for idx, row in valid_df.iterrows():
                    if prompt_template:
                        # Build kwargs for the prompt template
                        prompt_kwargs = {}
                        for col in columns_to_use:
                            if col in df.columns:
                                prompt_kwargs[col] = str(row[col])

                        # Add category_list if labels provided
                        if labels:
                            # Format labels as they would appear in the prompt
                            category_list = "\n".join(labels)
                            prompt_kwargs["category_list"] = category_list

                        # Generate the full prompt for this row
                        try:
                            full_prompt = prompt_template(**prompt_kwargs)
                            prompt_tokens = estimate_tokens(full_prompt, model)

                            # Store first prompt as sample
                            if sample_prompt is None:
                                sample_prompt = full_prompt
                                console.print(
                                    f"[dim]Sample prompt token count: {prompt_tokens}[/dim]"
                                )
                        except Exception as e:
                            console.print(
                                f"[red]Warning: Could not generate prompt for row {idx}: {e}[/red]"
                            )
                            # Fallback to simple concatenation
                            text_content = " ".join(
                                str(row[col])
                                for col in columns_to_use
                                if col in df.columns
                            )
                            prompt_tokens = estimate_tokens(text_content, model)
                    else:
                        # Simple case: just the text content plus system prompt
                        text_content = " ".join(
                            str(row[col]) for col in columns_to_use if col in df.columns
                        )
                        prompt_tokens = estimate_tokens(text_content, model)

                        if include_context:
                            prompt_tokens += system_tokens

                    total_input_tokens += prompt_tokens
                    rows_processed += 1

                    # Calculate output tokens for rows processed so far
                    total_output_tokens = (
                        rows_processed * expected_output_tokens_per_row
                    )

                    # Update progress bar
                    progress.update(task, advance=1)

                    # Update live display every 10 rows or on last row
                    if rows_processed % 10 == 0 or rows_processed == num_rows:
                        live.update(
                            create_display_table(
                                total_input_tokens, total_output_tokens, rows_processed
                            )
                        )
    else:
        # Original non-progress version
        for idx, row in valid_df.iterrows():
            if prompt_template:
                # Build kwargs for the prompt template
                prompt_kwargs = {}
                for col in columns_to_use:
                    if col in df.columns:
                        prompt_kwargs[col] = str(row[col])

                # Add category_list if labels provided
                if labels:
                    # Format labels as they would appear in the prompt
                    category_list = "\n".join(labels)
                    prompt_kwargs["category_list"] = category_list

                # Generate the full prompt for this row
                try:
                    full_prompt = prompt_template(**prompt_kwargs)
                    prompt_tokens = estimate_tokens(full_prompt, model)

                    # Store first prompt as sample
                    if sample_prompt is None:
                        sample_prompt = full_prompt
                        print(f"Sample prompt token count: {prompt_tokens}")
                except Exception as e:
                    print(f"Warning: Could not generate prompt for row {idx}: {e}")
                    # Fallback to simple concatenation
                    text_content = " ".join(
                        str(row[col]) for col in columns_to_use if col in df.columns
                    )
                    prompt_tokens = estimate_tokens(text_content, model)
            else:
                # Simple case: just the text content plus system prompt
                text_content = " ".join(
                    str(row[col]) for col in columns_to_use if col in df.columns
                )
                prompt_tokens = estimate_tokens(text_content, model)

                if include_context:
                    prompt_tokens += system_tokens

            total_input_tokens += prompt_tokens

    # Calculate the total output tokens from dataset
    total_output_tokens = num_rows * expected_output_tokens_per_row

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * prices["input"]
    output_cost = (total_output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    if show_progress:
        console.print("\n[bold green]✓ Cost estimation complete![/bold green]")

    # Return CostPlanResult dataclass
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
