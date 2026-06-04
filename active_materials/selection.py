"""Feature selection and dimensionality reduction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class FeatureSelectionResult:
    """Feature selection output summary."""

    output_path: Path
    selected_features_path: Path
    method: str
    input_feature_count: int
    output_feature_count: int


def reduce_features(
    *,
    input_csv: Path,
    output_csv: Path,
    n_features: int,
    method: str = "variance",
    id_column: str | int = 0,
    target_column: str | int | None = None,
    exclude_columns: list[str | int] | None = None,
    random_seed: int = 1007,
    selected_features_output: Path | None = None,
) -> FeatureSelectionResult:
    """Select or reduce features and write a new CSV."""

    data = pd.read_csv(input_csv)
    sample_ids = _select_column(data, id_column)
    excluded = {_column_name(data, id_column)}
    if target_column is not None:
        excluded.add(_column_name(data, target_column))
    for column in exclude_columns or []:
        excluded.add(_column_name(data, column))

    feature_frame = data.drop(columns=list(excluded))
    feature_frame = feature_frame.select_dtypes(include=[np.number])
    if feature_frame.empty:
        raise ValueError("No numeric feature columns found.")
    if n_features <= 0:
        raise ValueError("n_features must be greater than zero.")

    method = method.lower()
    if method == "pca":
        reduced, selected_names = _pca(feature_frame, n_features)
    else:
        selected_names = _select_feature_names(
            feature_frame=feature_frame,
            target=_target_values(data, target_column),
            n_features=n_features,
            method=method,
            random_seed=random_seed,
        )
        reduced = feature_frame.loc[:, selected_names]

    output = pd.concat(
        [pd.DataFrame({sample_ids.name: sample_ids}), reduced.reset_index(drop=True)],
        axis=1,
    )
    if target_column is not None:
        target = _select_column(data, target_column).reset_index(drop=True)
        output[target.name] = target

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    selected_features_output = selected_features_output or output_csv.with_name(
        f"{output_csv.stem}_selected_features.txt"
    )
    selected_features_output.parent.mkdir(parents=True, exist_ok=True)
    selected_features_output.write_text("\n".join(selected_names) + "\n", encoding="utf-8")

    return FeatureSelectionResult(
        output_path=output_csv,
        selected_features_path=selected_features_output,
        method=method,
        input_feature_count=feature_frame.shape[1],
        output_feature_count=len(selected_names),
    )


def _select_feature_names(
    *,
    feature_frame: pd.DataFrame,
    target: np.ndarray | None,
    n_features: int,
    method: str,
    random_seed: int,
) -> list[str]:
    if method == "variance":
        scores = feature_frame.var(axis=0).sort_values(ascending=False)
        return scores.head(n_features).index.tolist()

    if method == "correlation":
        return _correlation_filtered_features(feature_frame, n_features)

    if method == "random_forest":
        if target is None:
            raise ValueError("target_column is required for random_forest feature selection.")
        model = RandomForestRegressor(n_estimators=300, random_state=random_seed, n_jobs=-1)
        model.fit(feature_frame, target)
        scores = pd.Series(model.feature_importances_, index=feature_frame.columns)
        return scores.sort_values(ascending=False).head(n_features).index.tolist()

    if method == "mutual_info":
        if target is None:
            raise ValueError("target_column is required for mutual_info feature selection.")
        scores = pd.Series(
            mutual_info_regression(feature_frame, target, random_state=random_seed),
            index=feature_frame.columns,
        )
        return scores.sort_values(ascending=False).head(n_features).index.tolist()

    available = "variance, correlation, random_forest, mutual_info, pca"
    raise ValueError(f"Unknown feature selection method '{method}'. Available methods: {available}")


def _correlation_filtered_features(feature_frame: pd.DataFrame, n_features: int) -> list[str]:
    ranked = feature_frame.var(axis=0).sort_values(ascending=False).index.tolist()
    corr = feature_frame.corr().abs()
    selected: list[str] = []
    for candidate in ranked:
        if len(selected) >= n_features:
            break
        if all(corr.loc[candidate, chosen] < 0.9 for chosen in selected):
            selected.append(candidate)
    if len(selected) < n_features:
        for candidate in ranked:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= n_features:
                break
    return selected


def _pca(feature_frame: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, list[str]]:
    n_components = min(n_components, feature_frame.shape[1], feature_frame.shape[0])
    scaled = StandardScaler().fit_transform(feature_frame)
    pca = PCA(n_components=n_components, random_state=1007)
    values = pca.fit_transform(scaled)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(values, columns=columns), columns


def _target_values(data: pd.DataFrame, target_column: str | int | None) -> np.ndarray | None:
    if target_column is None:
        return None
    return _select_column(data, target_column).to_numpy(dtype=float)


def _select_column(data: pd.DataFrame, column: str | int) -> pd.Series:
    if isinstance(column, int):
        return data.iloc[:, column]
    return data[column]


def _column_name(data: pd.DataFrame, column: str | int) -> str:
    if isinstance(column, int):
        return str(data.columns[column])
    return column
