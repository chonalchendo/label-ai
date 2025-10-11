import typing as T

import polars as pl
import pydantic as pdt

import labelai.datasets as datasets
import labelai.jobs.base as base


class PreprocessingJob(base.Job):
    KIND: T.Literal["preprocessing"] = "preprocessing"

    input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        df = self.input.read()

        df = df.with_columns(
            [
                pl.col("title")
                .map_elements(self._clean_title, return_dtype=pl.Utf8)
                .alias("title"),
                pl.col("abstract")
                .map_elements(self._clean_abstract, return_dtype=pl.Utf8)
                .alias("abstract"),
            ]
        )

        self.output.write(df)

    def _clean_abstract(self, text: str) -> str:
        if not text:
            return ""
        paragraphs = text.split("\n\n")
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            cleaned_paragraph = " ".join(line.strip() for line in paragraph.split("\n"))
            cleaned_paragraphs.append(cleaned_paragraph)
        return "\n\n".join(cleaned_paragraphs)

    def _clean_title(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(line.strip() for line in text.split("\n"))
