from pathlib import Path

import torch
from FlagEmbedding import BGEM3FlagModel
from rich import print
from sklearn.model_selection import train_test_split

from labelai import (
    compute_and_save_labels,
    format_taxonomy,
    get_label_embeddings,
    get_prompt_template,
    load_dataset,
    load_labels,
    load_taxonomy,
    plan_cost,
)

RANDOM_STATE = 42
TRAIN_SIZE = 20_000
TEST_SIZE = 500
EMBEDDING_MODEL = "BAAI/bge-m3"


if __name__ == "__main__":
    df = load_dataset()
    labels = load_taxonomy()

    print(df)
    print(labels)

    # This will randomly split your data
    train_records, test_records = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=42,  # for reproducibility
    )

    labels_example = [format_taxonomy(path) for path in labels[:10]]

    # Estimate cost with the prompt template
    cost = plan_cost(
        df=train_records,
        text_column="abstract",
        additional_columns=["title"],  # Include title in the prompt
        model="gpt-4o-mini",
        labels=labels_example,
        prompt_template=get_prompt_template,
        expected_output_tokens_per_row=10,  # Override since output is just an ID like "medtop:20000842"
    )

    print(cost)

    # Additional insights
    print(f"\nPer-record cost: ${cost.total_cost_usd / cost.num_rows:.6f}")
    print(f"Cost per 1000 records: ${(cost.total_cost_usd / cost.num_rows) * 1000:.2f}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model on GPU
    model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)

    # Load taxonomy and create mappings
    leaf_paths = load_taxonomy()
    label_to_code = {format_taxonomy(path): path[-1][0] for path in leaf_paths}
    code_to_label = {path[-1][0]: format_taxonomy(path) for path in leaf_paths}
    labels = [format_taxonomy(path) for path in leaf_paths]

    # Define the Google Drive directory
    folder = Path("data/embeddings")

    # Ensure the directory exists
    folder.mkdir(parents=True, exist_ok=True)

    label_embeddings_path = folder / "label_embeddings.npy"
    # Define paths for records with top labels
    train_top_labels_pkl = folder / "train_records_with_top_labels.pkl"
    test_top_labels_pkl = folder / "test_records_with_top_labels.pkl"

    label_embeddings = get_label_embeddings(
        labels=labels, model=model, device=device, embeddings_path=label_embeddings_path
    )

    # Process training data with explicit caching logic
    if train_top_labels_pkl.exists():
        print("Loading cached training labels...")
        train_df = load_labels(train_top_labels_pkl)
    else:
        print("Computing training labels...")
        train_df = compute_and_save_labels(
            df=train_records,
            path=train_top_labels_pkl,
            label_embeddings=label_embeddings,
            model=model,
            labels=labels,
            label_to_code=label_to_code,
            device=device,
        )

    # Process test data with explicit caching logic
    if test_top_labels_pkl.exists():
        print("Loading cached test labels...")
        test_df = load_labels(test_top_labels_pkl)
    else:
        print("Computing test labels...")
        test_df = compute_and_save_labels(
            df=test_records,
            path=test_top_labels_pkl,
            label_embeddings=label_embeddings,
            model=model,
            labels=labels,
            label_to_code=label_to_code,
            device=device,
        )

    print(f"\n✓ Complete: {len(train_df)} train, {len(test_df)} test records")

    # Get the first row regardless of index
    first_row = train_df.iloc[0]
    print("Sample train record:")
    print(f"Title: {first_row['title']}")
    print("Top 5 labels:")
    for code, label in first_row["top_labels"][:5]:
        print(f"  - [{code}] {label}")
