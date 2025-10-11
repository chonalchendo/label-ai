import typing as T

import pandas as pd
import polars as pl
import pydantic as pdt

import labelai.datasets as datasets
import labelai.jobs.base as base


class TaxonomyJob(base.Job):
    KIND: T.Literal["taxonomy"] = "taxonomy"

    input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    output_labels: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_label_to_code: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        df = self.input.read()
        df = df.to_pandas()
        dff = self._transform_taxonomy(df)
        label_to_code = {
            self._format_taxonomy(path): path[-1][0]
            for path in dff["taxonomy_path"].to_list()
        }
        labels = [
            self._format_taxonomy(path) for path in dff["taxonomy_path"].to_list()
        ]

        output_df = pl.DataFrame({"taxonomy_label": labels})

        self.output_labels.write(output_df)
        self.output_label_to_code.write(label_to_code)

        return locals()

    def _transform_taxonomy(self, df: pd.DataFrame) -> pl.DataFrame:
        data = df.copy()
        level_cols = [f"Level{i}/NewsCode" for i in range(1, 7)]

        path = []
        leaf_paths = []

        for i in range(data.shape[0]):
            row = data.iloc[i]

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

            is_leaf = self._check_is_leaf(
                df=data, level_cols=level_cols, i=i, current_level=current_level
            )

            # If it's a leaf, store the current path
            if is_leaf:
                # formatted_path = _format_taxonomy(path)
                leaf_paths.append(path.copy())

        data = {"taxonomy_path": leaf_paths}

        return pl.DataFrame(data)

    def _check_is_leaf(
        self, df: pd.DataFrame, level_cols: list[str], i: int, current_level: int
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

    def _format_taxonomy(self, path: list[str]) -> str:
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
