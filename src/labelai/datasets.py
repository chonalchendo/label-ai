import json
import os
from io import BytesIO
from pathlib import Path
from typing import Sequence

import httpx
import kagglehub
import pandas as pd
from rich import print

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
        print(f"Path: {path} exists. Loading dataset.")
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
) -> list[str]:
    output_path = Path("data") / "taxonomy_labels.json"
    if output_path.exists():
        with open(output_path, "r") as f:
            taxonomy = json.load(f)
            return taxonomy

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the file
    response = httpx.get(url)

    # Create a file-like object from the downloaded content
    excel_data = BytesIO(response.content)

    # Load the spreadsheet, skipping the first row and using the second row as headers
    df = pd.read_excel(excel_data, skiprows=1)
    df = df[columns]

    taxonomy = _transform_taxonomy(df)

    with open(output_path, "w") as f:
        f.write(json.dumps(taxonomy))

    return taxonomy


def _transform_taxonomy(df: pd.DataFrame) -> list[str]:
    level_cols = [f"Level{i}/NewsCode" for i in range(1, 7)]

    path = []
    leaf_paths = []

    for i in range(df.shape[0]):
        row = df.iloc[i]

        # Determine the level by finding the non-empty level column
        non_empty_levels = row[level_cols].notna()
        if non_empty_levels.sum() != 1:
            # Skip rows with invalid level data (not exactly one level column filled)
            continue
        col = non_empty_levels.idxmax()
        current_level = level_cols.index(col) + 1  # Level number (1 to 6)

        # Extract the code, name, and definition
        qcode = row["NewsCode-QCode (flat)"]
        name = row["Name (en-US)"]
        definition = row["Definition (en-US)"]

        # Create a tuple for the current element
        element = (qcode, name, definition)

        # Update the path stack to match the current level
        while len(path) >= current_level:
            path.pop()
        path.append(element)

        is_leaf = _check_is_leaf(
            df=df, level_cols=level_cols, i=i, current_level=current_level
        )

        # If it's a leaf, store the current path
        if is_leaf:
            # formatted_path = _format_taxonomy(path)
            leaf_paths.append(path.copy())

    return leaf_paths


def format_taxonomy(path: list[str]) -> str:
    """
    Formats a taxonomy path into a string with names separated by '>' and the leaf definition in parentheses.
    Args:
        path (list of tuples): Each tuple contains (code, name, definition).
    Returns:
        str: Formatted string, e.g., 'name1 > name2 > name3 (definition3)'
    """
    names = [element[1] for element in path]
    joined_names = " > ".join(names)
    leaf_definition = path[-1][2]
    return f"{joined_names} ({leaf_definition})"


def _check_is_leaf(
    df: pd.DataFrame, level_cols: list[str], i: int, current_level: int
) -> bool:
    if i == len(df) - 1:
        return True

    next_row = df.iloc[i + 1]
    next_non_empty_levels = next_row[level_cols].notna()

    if next_non_empty_levels.sum() != 1:
        return True

    next_col = next_non_empty_levels.idxmax()
    next_level = level_cols.index(next_col) + 1

    if next_level <= current_level:
        return True

    return False
