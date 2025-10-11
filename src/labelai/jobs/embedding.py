import time
import typing as T
from pathlib import Path

import pandas as pd
import polars as pl
import pydantic as pdt
import torch
from FlagEmbedding import BGEM3FlagModel
from sklearn.model_selection import train_test_split

import labelai.datasets as datasets
import labelai.jobs.base as base


class EmbeddingJob(base.Job):
    KIND: T.Literal["embedding"] = "embedding"

    input_data: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    input_labels: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    input_labels_to_code: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    input_embeddings: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    output_label_embeddings: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_train_top_labels: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_test_top_labels: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    model: str
    train_size: int
    test_size: int
    shuffle: bool
    random_state: int

    @T.override
    def run(self) -> base.Locals:
        # 1. split data set into train and test
        input_df = self.input_data.read()
        input_df_pandas = input_df.to_pandas()
        train_df, test_df = train_test_split(
            input_df_pandas,
            train_size=self.train_size,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        # 2. define the model and set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BGEM3FlagModel(self.model, use_fp16=True)

        # 3. get the label embeddings
        labels_df = self.input_labels.read()
        labels = labels_df["taxonomy_label"].to_list()

        label_to_code = self.input_labels_to_code.read()

        if Path(self.output_label_embeddings.path).exists():
            print("Loading cached label embeddings...")
            label_embeddings = self.input_embeddings.read()
            print(f"✓ Loaded {len(labels)} label embeddings from cache")
        else:
            start_time = time.perf_counter()
            print("Getting label embeddings...")
            embeddings_np = model.encode(
                labels_df["taxonomy_label"].to_list(), batch_size=32
            )["dense_vecs"]
            label_embeddings = torch.from_numpy(embeddings_np).to(device)
            self.output_label_embeddings.write(label_embeddings)
            end_time = time.perf_counter()
            print(
                f"✓ Embedded {len(labels)} labels in {end_time - start_time:.2f} seconds"
            )

        # 4. attach the top labels to each record in the train and test datasets
        if not Path(self.output_train_top_labels.path).exists():
            print("Attaching top labels to training records...")
            train_records = self._attach_top_labels(
                df=train_df,
                label_embeddings=label_embeddings,
                model=model,
                labels=labels,
                label_to_code=label_to_code,
                device=device,
            )
            self.output_train_top_labels.write(pl.from_pandas(train_records))

        if not Path(self.output_test_top_labels.path).exists():
            print("Attaching top labels to test records...")
            test_records = self._attach_top_labels(
                df=test_df,
                label_embeddings=label_embeddings,
                model=model,
                labels=labels,
                label_to_code=label_to_code,
                device=device,
            )
            self.output_test_top_labels.write(pl.from_pandas(test_records))

        return locals()

    def _attach_top_labels(
        self,
        df: "pd.DataFrame",
        label_embeddings: "torch.Tensor",
        model: BGEM3FlagModel,  # Encoder model
        labels: list[str],
        label_to_code: dict[str, str],
        device: "torch.device",
        top_k: int = 100,
        batch_size: int = 128,
    ) -> pd.DataFrame:
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
            doc_embeddings_np = model.encode(batch_documents, batch_size=32)[
                "dense_vecs"
            ]
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
