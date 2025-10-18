import os
import typing as T
from pathlib import Path

import polars as pl
import pydantic as pdt
from dotenv import load_dotenv
from openai import AsyncOpenAI

import labelai.cost as cost
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

    # cost parameters
    max_cost: float
    expected_output_token_length: int
    models_csv_path: str

    @T.override
    async def run(self) -> base.Locals:
        # 1. load datasets
        train_top_labels = self.input_train_top_labels.read()
        test_top_labels = self.input_test_top_labels.read()

        # 2. calculate cost of labelling each dataframe
        cost_train_df = cost.plan_cost_v2(
            data=train_top_labels,
            models=self.models,
            template_path=Path(self.template_path),
            label_column=self.label_column,
            context_columns=self.context_columns,
            expected_output_token_length=self.expected_output_token_length,
            models_csv_path=self.models_csv_path,
        )

        cost_test_df = cost.plan_cost_v2(
            data=test_top_labels,
            models=self.models,
            template_path=Path(self.template_path),
            label_column=self.label_column,
            context_columns=self.context_columns,
            expected_output_token_length=self.expected_output_token_length,
            models_csv_path=self.models_csv_path,
        )

        total_cost_usd = round(
            cost_train_df["total_cost_usd"].sum()
            + cost_test_df["total_cost_usd"].sum(),
            2,
        )
        print(f"Total Labelling Cost: {total_cost_usd}")

        if total_cost_usd > self.max_cost:
            print(
                f"Cost of labelling dataset exceeds specified max cost:\n- Max Cost: {self.max_cost}\n- Total Cost: {total_cost_usd}"
            )
            raise ValueError("Labelling too expensive")

        # 3. initialise client
        client = AsyncOpenAI(
            base_url=self.base_url, api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # 4. label datasets
        print("Labelling the training dataset...")
        train_df = await labeler.label_dataset(
            df=train_top_labels.to_pandas(),
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
            df=test_top_labels.to_pandas(),
            client=client,
            models=self.models,
            template_path=Path(self.template_path),
            context_columns=self.context_columns,
            label_column=self.label_column,
            batch_size=self.batch_size,
            majority_threshold=self.majority_threshold,
        )

        # 5. write out the results
        self.output_train_df.write(pl.from_pandas(train_df))
        self.output_test_df.write(pl.from_pandas(test_df))

        return locals()
