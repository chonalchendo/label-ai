import typing as T

import pydantic as pdt
from rich import print

import labelai.datasets as datasets
import labelai.jobs.base as base


class ExtractJob(base.Job):
    KIND: T.Literal["extract"] = "extract"

    data_input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    data_output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    taxonomy_input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    taxonomy_output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        # load raw data
        print(f"Loading raw data from: {self.data_input.path}")
        raw_data = self.data_input.read()

        print(f"Loading raw taxonomy from: {self.taxonomy_input.path}")
        raw_taxonomy = self.taxonomy_input.read()

        # output raw data
        print(f"Writing raw data to: {self.data_output.path}")
        self.data_output.write(raw_data)

        print(f"Writing raw taxonomy data to: {self.taxonomy_output.path}")
        self.taxonomy_output.write(raw_taxonomy)

        return locals()
