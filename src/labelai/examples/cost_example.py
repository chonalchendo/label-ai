import pandas as pd

from labelai import plan_cost

# Example usage
if __name__ == "__main__":
    # Create sample dataset
    sample_data = pd.DataFrame(
        {
            "text": [
                "This product is amazing! Best purchase ever.",
                "Terrible quality, would not recommend.",
                "It's okay, nothing special but does the job.",
                "Exceeded my expectations in every way.",
                "Complete waste of money, broke after one day.",
            ],
            "product_id": [1, 2, 3, 4, 5],
        }
    )

    # Define system prompt for classification task
    system_prompt = """You are a sentiment classifier. 
    Classify the given text as one of the provided labels.
    Respond with only the classification label."""

    # Define possible labels for categorization
    labels = ["positive", "negative", "neutral"]

    # Estimate cost with automatic output token calculation from labels
    cost_estimate = plan_cost(
        df=sample_data,
        text_column="text",
        model="gpt-4o-mini",
        labels=labels,  # Will automatically calculate expected output tokens
        system_prompt=system_prompt,
        include_context=True,
    )

    # Print results
    print("Cost Estimation Results:")
    print(f"Total Cost: ${cost_estimate.total_cost_usd}")
    print(f"Input Tokens: {cost_estimate.input_tokens:,}")
    print(f"Output Tokens: {cost_estimate.output_tokens:,}")
    print(f"Number of rows: {cost_estimate.num_rows}")
    print(f"Average input tokens per row: {cost_estimate.avg_input_tokens_per_row:.1f}")
    print(f"Breakdown:")
    print(f"  - Input cost: ${cost_estimate.input_cost_usd}")
    print(f"  - Output cost: ${cost_estimate.output_cost_usd}")

    print("\n" + "=" * 50 + "\n")

    # Example with longer labels
    detailed_labels = [
        "strongly positive with high confidence",
        "moderately positive with medium confidence",
        "neutral or unclear sentiment",
        "moderately negative with medium confidence",
        "strongly negative with high confidence",
    ]

    cost_estimate_detailed = plan_cost(
        df=sample_data,
        text_column="text",
        model="gpt-4o-mini",
        labels=detailed_labels,
        system_prompt=system_prompt,
        include_context=True,
    )

    print("Cost Estimation with Detailed Labels:")
    print(f"Total Cost: ${cost_estimate_detailed.total_cost_usd}")
    print(
        f"Output Tokens: {cost_estimate_detailed.output_tokens:,} (higher due to longer labels)"
    )
