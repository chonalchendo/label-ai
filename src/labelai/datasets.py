import abc
import json
import os
import typing as T
from io import BytesIO
from pathlib import Path

import httpx
import kagglehub
import polars as pl
import pydantic as pdt
import torch
from rich import print

T_Read = T.TypeVar("T_Read")
T_Write = T.TypeVar("T_Write")


class Reader(
    abc.ABC, pdt.BaseModel, T.Generic[T_Read], strict=True, frozen=False, extra="forbid"
):
    """Base class for all readers."""

    KIND: str
    path: str
    columns: T.Sequence[str] | None = None
    limit: int | None = None

    @abc.abstractmethod
    def read(self) -> T_Read:
        """Read the data from the path."""
        pass

    def _http_download(self) -> BytesIO:
        resp = httpx.get(self.path)
        return BytesIO(resp.content)

    def _kaggle_path(self) -> str:
        handle_ = "/".join(self.path.split("/")[:-1])
        file_name = self.path.split("/")[-1]
        handle = kagglehub.dataset_download(handle_)
        return os.path.join(handle, file_name)


class ExcelReader(Reader[pl.DataFrame]):
    KIND: T.Literal["excel"] = "excel"

    @T.override
    def read(self) -> pl.DataFrame:
        path = self.path
        if self.path.startswith("https"):
            path = self._http_download()

        df = pl.read_excel(path, columns=self.columns, read_options={"header_row": 1})
        print(df)
        return df


class JSONToDictReader(Reader):
    KIND: T.Literal["json_dict"] = "json_dict"

    @T.override
    def read(self) -> dict:
        with open(self.path, "r") as f:
            data = json.loads(f.read())
            return data


class JSONReader(Reader[pl.DataFrame]):
    KIND: T.Literal["json"] = "json"

    # output_dir: str
    kaggle: bool = False

    @T.override
    def read(self) -> pl.DataFrame:
        path = self.path

        # Download and read from Kaggle
        if self.kaggle:
            path = self._kaggle_path()

        # Read JSON with Polars (try regular then NDJSON format)
        try:
            df = pl.read_json(path)
        except Exception as e:
            print(f"Failed to read as regular JSON: {e}")
            print("Attempting to read as NDJSON (newline-delimited JSON)")
            df = pl.read_ndjson(path)

        print(df)
        return df


class ParquetReader(Reader[pl.DataFrame]):
    KIND: T.Literal["parquet"] = "parquet"

    @T.override
    def read(self) -> pl.DataFrame:
        df = pl.read_parquet(self.path)
        if self.limit:
            return df.limit(self.limit)
        return df


class CSVReader(Reader[pl.DataFrame]):
    KIND: T.Literal["csv"] = "csv"

    @T.override
    def read(self) -> pl.DataFrame:
        df = pl.read_csv(self.path)
        return df


class TorchReader(Reader[torch.Tensor]):
    KIND: T.Literal["torch"] = "torch"

    @T.override
    def read(self, map_location: T.Optional["torch.device"] = None) -> "torch.Tensor":
        """Load a PyTorch tensor from a .pt file."""
        tensor = torch.load(self.path, map_location=map_location)
        return tensor


class Writer(
    abc.ABC,
    pdt.BaseModel,
    T.Generic[T_Write],
    strict=True,
    frozen=False,
    extra="forbid",
):
    """Base class for all writers."""

    KIND: str
    path: str

    @abc.abstractmethod
    def write(self, data: T_Write) -> None:
        """Write the data to the path."""
        pass

    def _ensure_parent_dir(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)


class ParquetWriter(Writer[pl.DataFrame]):
    KIND: T.Literal["parquet"] = "parquet"

    @T.override
    def write(self, data: pl.DataFrame) -> None:
        file_ext = self.path.split(".")[-1]
        if file_ext != "parquet":
            raise ValueError("File extension is not parquet.")

        self._ensure_parent_dir(self.path)
        data.write_parquet(self.path)


class TorchWriter(Writer[torch.Tensor]):
    KIND: T.Literal["torch"] = "torch"

    @T.override
    def write(self, data: "torch.Tensor") -> None:
        """Save a PyTorch tensor to a .pt file."""
        self._ensure_parent_dir(self.path)
        torch.save(data.cpu(), self.path)


class CSVWriter(Writer[pl.DataFrame]):
    KIND: T.Literal["csv"]

    @T.override
    def write(self, data: pl.DataFrame) -> None:
        self._ensure_parent_dir(self.path)
        data.write_csv(self.path)


class JSONWriter(Writer):
    KIND: T.Literal["json"] = "json"

    @T.override
    def write(self, data: dict) -> None:
        self._ensure_parent_dir(self.path)
        with open(self.path, "w") as f:
            f.write(json.dumps(data))


ReaderKind = (
    JSONReader
    | JSONToDictReader
    | ExcelReader
    | ParquetReader
    | TorchReader
    | CSVReader
)
WriterKind = ParquetWriter | TorchWriter | JSONWriter | CSVWriter
