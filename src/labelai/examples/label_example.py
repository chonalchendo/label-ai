import os

import pandas as pd
import uvloop
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich import print

from labelai import label_dataset

load_dotenv()

MODEL = "gpt-4o-mini"
LABELS = ["bug", "feature", "question"]

if __name__ == "__main__":
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
