"""Feature matrix helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from active_materials.feature_utils import FeatureExtract


def build_feature_matrix(
    data: pd.DataFrame,
    *,
    feature_columns: list[str | int],
    structure_path_column: str | int | None = None,
    use_structure_features: bool = False,
) -> np.ndarray:
    """Build a model-ready feature matrix from tabular and optional structure features."""

    parts: list[np.ndarray] = []
    if feature_columns:
        parts.append(_select_columns(data, feature_columns).to_numpy(dtype=float))

    if use_structure_features:
        if structure_path_column is None:
            raise ValueError("structure_path_column is required when use_structure_features is true.")
        structures = [
            Structure.from_file(str(Path(path)))
            for path in _select_column(data, structure_path_column).tolist()
        ]
        structure_features = FeatureExtract().get_features(structures).to_numpy(dtype=float)
        parts.append(structure_features)

    if not parts:
        raise ValueError("At least one of feature_columns or use_structure_features must be provided.")

    return np.hstack(parts) if len(parts) > 1 else parts[0]


def select_target(data: pd.DataFrame, target_column: str | int) -> np.ndarray:
    """Return the supervised target vector."""

    return _select_column(data, target_column).to_numpy(dtype=float)


def select_sample_ids(data: pd.DataFrame, sample_id_column: str | int) -> pd.Series:
    """Return the sample identifier column."""

    return _select_column(data, sample_id_column)


def _select_columns(data: pd.DataFrame, columns: list[str | int]) -> pd.DataFrame:
    if all(isinstance(column, int) for column in columns):
        return data.iloc[:, columns]
    return data.loc[:, columns]


def _select_column(data: pd.DataFrame, column: str | int) -> pd.Series:
    if isinstance(column, int):
        return data.iloc[:, column]
    return data[column]
