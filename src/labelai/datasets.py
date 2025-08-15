import json
import os
from io import BytesIO
from pathlib import Path
from typing import Sequence

import httpx
import kagglehub
import pandas as pd

# Columns to select
TAXONOMY_COLUMNS = [
    "NewsCode-URI",
    "NewsCode-QCode (flat)",
    "Level1/NewsCode",
    "Level2/NewsCode",
    "Level3/NewsCode",
    "Level4/NewsCode",
    "Level5/NewsCode",
    "Level6/NewsCode",
    "Name (en-US)",
    "Definition (en-US)",
]


def clean_abstract(text: str) -> str:
    """Clean the abstract by preserving paragraphs and removing unnecessary newlines within paragraphs."""
    # Split into paragraphs using double newlines
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Replace single newlines with spaces within each paragraph
        cleaned_paragraph = " ".join(line.strip() for line in paragraph.split("\n"))
        cleaned_paragraphs.append(cleaned_paragraph)
    # Join paragraphs back with double newlines
    return "\n\n".join(cleaned_paragraphs)


def clean_title(text: str) -> str:
    """Clean the title by removing extra spaces and joining lines with spaces."""
    return " ".join(line.strip() for line in text.split("\n"))


def load_dataset(
    path: str = "Cornell-University/arxiv",
    file: str = "arxiv-metadata-oai-snapshot.json",
) -> pd.DataFrame:
    output_path = Path("data") / file.replace(".json", ".parquet")

    if not output_path.parent.exists():
        output_path.parent.mkdir()

    if output_path.exists():
        return pd.read_parquet(output_path)

    path = kagglehub.dataset_download(path)

    # Construct the full path to the JSON file
    file_path = os.path.join(path, file)

    records = []

    # Open the file and read the first two lines
    with open(file_path, "r") as file:
        for line in file:
            record: dict = json.loads(line)
            filtered_record = {
                "id": record.get("id", ""),
                "title": clean_title(record.get("title", "")),
                "abstract": clean_abstract(record.get("abstract", "")),
            }
            records.append(filtered_record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path)
    return df


def load_taxonomy(
    url: str = "https://www.iptc.org/std/NewsCodes/IPTC-MediaTopic-NewsCodes.xlsx",
    columns: Sequence[str] = TAXONOMY_COLUMNS,
) -> pd.DataFrame:
    # Download the file
    response = httpx.get(url)

    # Create a file-like object from the downloaded content
    excel_data = BytesIO(response.content)

    # Load the spreadsheet, skipping the first row and using the second row as headers
    df = pd.read_excel(excel_data, skiprows=1)

    return df[columns]
