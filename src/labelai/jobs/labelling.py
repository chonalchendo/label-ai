import os
import typing as T
from pathlib import Path

import polars as pl
import pydantic as pdt
from dotenv import load_dotenv
from openai import AsyncOpenAI

import labelai.datasets as datasets
import labelai.jobs.base as base
import labelai.labeler as labeler

# load environment variables from .env file
load_dotenv()


class LabellingJob(base.Job):
    KIND: T.Literal["labelling"] = "labelling"

    input_train_top_labels: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    input_test_top_labels: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    output_train_df: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_test_df: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    base_url: str
    models: list[str]
    template_path: str
    context_columns: list[str]
    label_column: str
    batch_size: int
    majority_threshold: int

    @T.override
    async def run(self) -> base.Locals:
        # 1. load datasets
        train_top_labels = self.input_train_top_labels.read().to_pandas()
        test_top_labels = self.input_test_top_labels.read().to_pandas()

        train_top_labels_sub = train_top_labels.iloc[:10]
        test_top_labels_sub = test_top_labels.iloc[:5]

        # 2. initialise client
        client = AsyncOpenAI(
            base_url=self.base_url, api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # 3. label datasets
        print("Labelling the training dataset...")
        train_df = await labeler.label_dataset(
            df=train_top_labels_sub,
            client=client,
            models=self.models,
            template_path=Path(self.template_path),
            context_columns=self.context_columns,
            label_column=self.label_column,
            batch_size=self.batch_size,
            majority_threshold=self.majority_threshold,
        )

        print("Labelling the test dataset...")
        test_df = await labeler.label_dataset(
            df=test_top_labels_sub,
            client=client,
            models=self.models,
            template_path=Path(self.template_path),
            context_columns=self.context_columns,
            label_column=self.label_column,
            batch_size=self.batch_size,
            majority_threshold=self.majority_threshold,
        )

        # 4. write out the results
        self.output_train_df.write(pl.from_pandas(train_df))
        self.output_test_df.write(pl.from_pandas(test_df))

        return locals()
