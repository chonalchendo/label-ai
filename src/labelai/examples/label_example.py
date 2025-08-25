import os

import pandas as pd
import uvloop
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich import print

from labelai import label_dataset

load_dotenv()

MODEL = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-chat-v3.1",
]
LABELS = ["bug", "feature", "question"]
BASE_URL = "https://openrouter.ai/api/v1"

if __name__ == "__main__":
    client = AsyncOpenAI(base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))

    # Create test dataset
    test_data = pd.DataFrame(
        {
            "message": [
                "The app crashes when I click submit",
                "Can you add dark mode?",
                "How do I reset my password?",
                "The loading spinner never stops",
                "It would be great to have CSV export",
            ]
        }
    )

    # Label the dataset
    labeled_df = uvloop.run(label_dataset(test_data, "message", LABELS, MODEL, client))

    # Show results
    print("\n[bold]Results:[/bold]")
    print(labeled_df[["message", "label"]])
    print(labeled_df)
