import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel


def get_label_embeddings(
    labels: list[str],
    model: BGEM3FlagModel,
    device: "torch.device",
    embeddings_path: "Path",
) -> "torch.Tensor":
    """Load or compute label embeddings."""

    # Try to load from cache
    if embeddings_path.exists():
        print("Loading cached label embeddings...")
        embeddings_np = np.load(embeddings_path)
        label_embeddings = torch.from_numpy(embeddings_np).to(device)
        print(f"✓ Loaded {len(labels)} label embeddings from cache")
        return label_embeddings

    # Compute embeddings
    print("Computing label embeddings...")
    start_time = time.perf_counter()
    label_embeddings_np = model.encode(labels, batch_size=32)["dense_vecs"]
    label_embeddings = torch.from_numpy(label_embeddings_np).to(device)
    end_time = time.perf_counter()
    print(f"✓ Embedded {len(labels)} labels in {end_time - start_time:.2f} seconds")

    # Save to cache
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, label_embeddings_np)
    print(f"✓ Saved embeddings to {embeddings_path}")

    return label_embeddings


def load_labels(pkl_path: "Path") -> pd.DataFrame:
    """Load DataFrame with labels from cache."""
    # return pickle.loads(pkl_path.read_bytes())
    return pd.read_pickle(pkl_path)


def compute_and_save_labels(
    df: "pd.DataFrame",
    path: Path,
    label_embeddings: "torch.Tensor",
    model: BGEM3FlagModel,  # Encoder model
    labels: list[str],
    label_to_code: dict[str, str],
    device: "torch.device",
    top_k: int = 100,
    batch_size: int = 128,
) -> "pd.DataFrame":
    """Compute top labels and save to cache."""
    df_with_labels = _attach_top_labels(
        df=df,
        label_embeddings=label_embeddings,
        model=model,
        labels=labels,
        label_to_code=label_to_code,
        device=device,
        top_k=top_k,
        batch_size=batch_size,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df_with_labels.to_pickle(path)
    print(f"Computed and saved to {path}")

    return df_with_labels


def _attach_top_labels(
    df: "pd.DataFrame",
    label_embeddings: "torch.Tensor",
    model: BGEM3FlagModel,  # Encoder model
    labels: list[str],
    label_to_code: dict[str, str],
    device: "torch.device",
    top_k: int = 100,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Attach top-k most similar labels to each record in the DataFrame.

    Args:
        df: DataFrame with 'title' and 'abstract' columns
        label_embeddings: Tensor of label embeddings (num_labels x embedding_dim)
        top_k: Number of top labels to retrieve per record
        batch_size: Number of records to process at once

    Returns:
        DataFrame with added 'top_labels' column containing list of (code, label) tuples
    """
    # Pre-allocate list for all top labels
    all_top_labels: list[list[tuple[str, str]]] = []

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        # Vectorized string concatenation
        batch_documents: list[str] = (
            "Title: "
            + batch_df["title"].astype(str)
            + "\nAbstract: "
            + batch_df["abstract"].astype(str)
        ).tolist()

        # Encode documents
        doc_embeddings_np = model.encode(batch_documents, batch_size=32)["dense_vecs"]
        doc_embeddings = torch.from_numpy(doc_embeddings_np).to(device)

        # Compute similarities
        similarity_matrix = torch.mm(doc_embeddings, label_embeddings.t())
        top_indices = (
            torch.topk(similarity_matrix, k=top_k, dim=1).indices.cpu().numpy()
        )

        # Batch create top labels
        batch_top_labels: list[list[tuple[str, str]]] = [
            [(label_to_code[labels[idx]], labels[idx]) for idx in indices]
            for indices in top_indices
        ]
        all_top_labels.extend(batch_top_labels)

    # Assign all at once
    df["top_labels"] = all_top_labels
    return df