PRICING = {
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    }  # per 1m tokens
}


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
