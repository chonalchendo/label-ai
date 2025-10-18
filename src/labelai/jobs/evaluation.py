import typing as T
from collections import Counter

import numpy as np
import polars as pl
import pydantic as pdt

import labelai.datasets as datasets
import labelai.jobs.base as base


class EvaluationJob(base.Job):
    KIND: T.Literal["evaluation"] = "evaluation"

    input_train_df: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    input_code_to_label: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    threshold: int

    @T.override
    def run(self) -> base.Locals:
        # 1. load in labelled samples
        train_df = self.input_train_df.read()

        # 2. Count the number of records with a majority vote and those with not
        labels = train_df.filter(pl.col("majority_vote") != "NO_CONCENSUS")[
            "majority_vote"
        ].to_list()

        labels_no_concensus_count = train_df.filter(
            pl.col("majority_vote") == "NO_CONCENSUS"
        ).shape[0]

        unlabelled_rows_pct = (
            round(labels_no_concensus_count / train_df.shape[0], 2) * 100
        )

        # 4. Calculate the max, min, avg number of documents per class.
        # 5. Calculate the number of classes with fewer than n documents
        label_counts = Counter(labels)
        counts = list(label_counts.values())
        num_classes = len(label_counts)
        avg_docs_per_class = max_docs = min_docs = classes_with_less_than_threshold = 0

        if num_classes > 0:
            avg_docs_per_class = np.mean(counts)
            max_docs = np.max(counts)
            min_docs = np.min(counts)
            classes_with_less_than_threshold = sum(
                1 for count in counts if count < self.threshold
            )

        print(f"Average number of documents per class: {avg_docs_per_class:.2f}")
        print(f"Maximum number of documents in a class: {max_docs}")
        print(f"Minimum number of documents in a class: {min_docs}")
        print(
            f"Number of classes with fewer than {self.threshold} documents: {classes_with_less_than_threshold}"
        )
        print(f"Percentage of records with NO_CONCENSUS: {unlabelled_rows_pct}")

        # 6. Get the most common labels
        code_to_label = self.input_code_to_label.read()
        labels_df = train_df.filter(
            pl.col("majority_vote") != "NO_CONCENSUS"
        ).with_columns(
            pl.col("majority_vote").replace(code_to_label).alias("label_title")
        )

        label_freq_df = labels_df["label_title"].value_counts(sort=True)
        print(label_freq_df)
