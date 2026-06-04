"""Main active-learning workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from active_materials.config import PipelineConfig
from active_materials.features import build_feature_matrix, select_sample_ids, select_target
from active_materials.strategies import get_strategy

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SelectionResult:
    """Summary of one active-learning selection."""

    selected: pd.DataFrame
    metrics: dict[str, float]


class ActiveLearningPipeline:
    """Train a model on seed data and select the next unlabeled batch."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> SelectionResult:
        """Run one active-learning selection round."""

        seed = pd.read_csv(self.config.seed_csv)
        unlabeled = pd.read_csv(self.config.unlabeled_csv)
        train = self._prepare_training_data(seed)
        x_pool = self._prepare_unlabeled_features(unlabeled, train["scaler"])
        estimator = self._build_estimator(train["x_train"])
        estimator.fit(train["x_train"], train["y_train"])
        metrics = self._evaluate(estimator, train)

        strategy = partial(
            get_strategy(self.config.strategy),
            y_training=train["y_train"],
        )
        selected_idx, _ = strategy(estimator, x_pool, number=self.config.batch_size)
        selected = self._build_selected_output(estimator, unlabeled, x_pool, selected_idx)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(self.config.selected_output_path, index=False)
        pd.DataFrame([metrics]).to_csv(self.config.metrics_output_path, index=False)
        LOGGER.info("Selected %s samples -> %s", len(selected), self.config.selected_output_path)
        LOGGER.info("Validation metrics -> %s", self.config.metrics_output_path)
        return SelectionResult(selected=selected, metrics=metrics)

    def _prepare_training_data(self, seed: pd.DataFrame) -> dict[str, np.ndarray]:
        x = build_feature_matrix(
            seed,
            feature_columns=self.config.feature_columns,
            structure_path_column=self.config.structure_path_column,
            use_structure_features=self.config.use_structure_features,
        )
        y = select_target(seed, self.config.target_column)
        if self.config.log_target:
            y = np.log(y)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
        )

        scaler = StandardScaler()
        if self.config.scale_features:
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        else:
            scaler.fit(x_train)
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
        }

    def _prepare_unlabeled_features(self, unlabeled: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
        x_pool = build_feature_matrix(
            unlabeled,
            feature_columns=self.config.feature_columns,
            structure_path_column=self.config.structure_path_column,
            use_structure_features=self.config.use_structure_features,
        )
        return scaler.transform(x_pool) if self.config.scale_features else x_pool

    def _build_estimator(self, x_train: np.ndarray) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(1.0, (1.0e-3, 1.0e3))
            * Matern([1.0] * x_train.shape[1], [[1.0e-3, 1.0e3]] * x_train.shape[1], 1.5)
            + WhiteKernel(1.0, (1.0e-3, 1.0e3))
        )
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=self.config.gp_normalize_y,
            random_state=self.config.random_seed,
            n_restarts_optimizer=self.config.gp_restarts,
            alpha=self.config.gp_alpha,
        )
        return estimator

    def _evaluate(self, estimator: GaussianProcessRegressor, train: dict[str, np.ndarray]) -> dict[str, float]:
        y_pred = estimator.predict(train["x_test"])
        return {
            "r2": float(r2_score(train["y_test"], y_pred)),
            "rmse": float(root_mean_squared_error(train["y_test"], y_pred)),
            "mae": float(mean_absolute_error(train["y_test"], y_pred)),
        }

    def _build_selected_output(
        self,
        estimator: GaussianProcessRegressor,
        unlabeled: pd.DataFrame,
        x_pool: np.ndarray,
        selected_idx: np.ndarray,
    ) -> pd.DataFrame:
        mu, std = estimator.predict(x_pool, return_std=True)
        selected = unlabeled.iloc[selected_idx].copy()
        selected.insert(0, "al_rank", np.arange(1, len(selected_idx) + 1))
        selected.insert(1, "al_sample_id", select_sample_ids(unlabeled, self.config.sample_id_column).iloc[selected_idx].values)
        selected["al_pred_mean"] = mu[selected_idx]
        selected["al_pred_std"] = std[selected_idx]
        return selected
