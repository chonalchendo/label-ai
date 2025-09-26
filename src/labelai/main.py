import os
from pathlib import Path

import uvloop
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich import print

from labelai.embeddings import load_labels
from labelai.labeler import label_dataset

load_dotenv()

FOLDER = Path("data/embeddings")
train_top_labels_pkl = FOLDER / "train_records_with_top_labels.pkl"

BASE_URL = "https://openrouter.ai/api/v1"
MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-chat-v3.1",
]


# Usage example:
async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))

    print("Loading cached training labels...")
    train_df = load_labels(train_top_labels_pkl)
    train_df = train_df.iloc[:5]

    results = await label_dataset(
        df=train_df,
        client=client,
        models=MODELS,
        template_path=Path("prompts/example_prompt.txt"),
        context_columns=["title", "abstract"],
        label_column="top_labels",
        batch_size=10,
    )

    print(results)

    # Save results
    results.to_csv("data/labeled_documents.csv", index=False)

    # Quick analysis
    print("\nModel Agreement Statistics:")
    print(results["agreement_pct"].describe())
    print("\nPredictions per model:")
    for model in MODELS:
        print(f"{model}: {results[f'{model}_prediction'].value_counts().head()}")


if __name__ == "__main__":
    uvloop.run(main())
