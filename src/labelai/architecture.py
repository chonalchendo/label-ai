from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass
class Model:
    name: str

    async def predict(self):
        pass


@dataclass
class DataReader:
    mode: Literal["pandas", "polars"]
    format: Literal["csv", "parquet", "json"]


@dataclass
class DataWriter:
    pass


@dataclass
class Taxonomy:
    pass


@dataclass
class Voter:
    pass


@dataclass
class Reviewer:
    threshold: float


@dataclass
class Embedder:
    pass

    def embed(self):
        pass


@dataclass
class Labeler:
    models: Sequence[Model]
    embedder: Embedder
    voter: Voter
    reviewer: Reviewer
    output: DataWriter

    async def label(self, data: DataReader):
        # 1. load the dataset into the specified mode from the specified format
        # 2. Sort out the taxonomy for the dataset to create labels
        # 3. Create embeddings to select top labels for each row
        # 4. Use ensemble of models to asynchronously assign a class to each record
        # 5. Use voter to evaluate how well the majority vote has done
        # 6. Use reviewer to understand what needs to be reviewed by a human -> output items to review as a dataset
        # 7. Generate report to understand class imbalance of labels
        # 8. Output labelled dataset to defined location and in defined format.
        pass
