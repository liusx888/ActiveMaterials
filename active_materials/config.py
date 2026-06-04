"""Configuration for active-learning runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PipelineConfig:
    """User-facing settings for one active-learning selection round."""

    seed_csv: Path = Path("data/seed.csv")
    unlabeled_csv: Path = Path("data/unlabeled.csv")
    output_dir: Path = Path("outputs")
    sample_id_column: str | int = 0
    feature_columns: list[str | int] = field(default_factory=list)
    target_column: str | int = "target"
    structure_path_column: str | int | None = None
    use_structure_features: bool = False
    batch_size: int = 30
    test_size: float = 0.2
    random_seed: int = 1007
    strategy: str = "expected_improvement"
    log_target: bool = False
    scale_features: bool = True
    gp_alpha: float = 1e-2
    gp_normalize_y: bool = False
    gp_restarts: int = 10
    selected_output: str = "selected.csv"
    metrics_output: str = "metrics.csv"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load settings from a YAML file."""

        with Path(path).open("r", encoding="utf-8") as file:
            values: dict[str, Any] = yaml.safe_load(file) or {}
        config = cls(**values)
        config._normalize_paths()
        return config

    def _normalize_paths(self) -> None:
        for field_name in ("seed_csv", "unlabeled_csv", "output_dir"):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, Path(value))

    @property
    def selected_output_path(self) -> Path:
        return self.output_dir / self.selected_output

    @property
    def metrics_output_path(self) -> Path:
        return self.output_dir / self.metrics_output
