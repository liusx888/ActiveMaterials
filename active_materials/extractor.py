"""Structure feature extraction command helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

from active_materials.feature_utils import FeatureExtract

LOGGER = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".cif", ".vasp", ".poscar")
DEFAULT_FILENAMES = ("POSCAR", "CONTCAR")
PAPER_FEATURE_COLUMNS = [
    "MASS_MEAN",
    "MASS_STD",
    "MASS_MIN",
    "ATOMIC_RADII_MEAN",
    "ATOMIC_RADII_STD",
    "ATOMIC_RADII_MAX",
    "ELECTRON_NEG_MEAN",
    "ELECTRON_NEG_MIN",
    "IONIZATION_ENERGIES_MEAN",
    "IONIZATION_ENERGIES_STD",
    "IONIZATION_ENERGIES_MIN",
    "ELECTRON_AFFINITIES_MEAN",
    "ELECTRON_AFFINITIES_STD",
    "ELECTRON_AFFINITIES_MIN",
    "IONIC_RADII_MEAN",
    "IONIC_RADII_STD",
    "IONIC_RADII_MIN",
    "VALENCE_MEAN",
    "VALENCE_STD",
    "VALENCE_MIN",
    "VALENCE_MODE",
    "SHELL_STD",
    "SHELL_MAX",
    "SHELL_MODE",
    "S_E_MEAN",
    "S_E_STD",
    "S_E_MAX",
    "P_E_MEAN",
    "P_E_MIN",
    "BLOCK_MEAN",
    "BLOCK_STD",
    "BLOCK_MAX",
    "S_UNFILLED_MEAN",
    "S_UNFILLED_STD",
    "S_UNFILLED_MODE",
    "P_UNFILLED_MEAN",
    "P_UNFILLED_STD",
    "P_UNFILLED_MIN",
    "P_UNFILLED_MODE",
    "D_UNFILLED_MEAN",
    "D_UNFILLED_STD",
    "D_UNFILLED_MIN",
    "D_UNFILLED_MODE",
    "MAX_OXIDATION_STATE_MEAN",
    "MAX_OXIDATION_STATE_STD",
    "MAX_OXIDATION_STATE_MAX",
    "MAX_OXIDATION_STATE_MIN",
    "MELTING_POINT_MEAN",
    "MELTING_POINT_STD",
    "MELTING_POINT_MAX",
    "MENDELEEV_NO_MEAN",
    "MENDELEEV_NO_MAX",
    "MENDELEEV_NO_MODE",
    "MOLAR_VOLUME_MIN",
    "MOLAR_VOLUME_MODE",
    "BOILING_POINT_RANGE",
    "ELEMENT_NUM",
    "L2_NORM",
    "N_ATOMS",
    "SPACE_GROUP",
    "GAMMA",
    "A",
    "C",
    "VOLUME_PER_ATOM",
    "DENSITY",
    "DISTANCE_STD",
    "DISTANCE_MAX",
]


@dataclass(slots=True)
class FeatureExtractionResult:
    """Feature extraction output summary."""

    output_path: Path
    total_files: int
    parsed_files: int
    skipped_files: int


def extract_features_from_directory(
    *,
    structure_dir: Path,
    output_path: Path,
    recursive: bool = False,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    id_column: str = "formula",
    select_columns: list[str] | None = None,
    use_default_feature_selection: bool = False,
) -> FeatureExtractionResult:
    """Extract composition and structure features from a directory of structure files."""

    structure_paths = find_structure_files(structure_dir, recursive=recursive, extensions=extensions)
    if not structure_paths:
        raise FileNotFoundError(f"No structure files found in {structure_dir}")

    sample_ids: list[str] = []
    structures: list[Structure] = []
    skipped = 0
    for path in structure_paths:
        try:
            structures.append(Structure.from_file(str(path)))
            sample_ids.append(path.name)
        except Exception as exc:
            skipped += 1
            LOGGER.warning("Skipped %s: %s", path, exc)

    if not structures:
        raise ValueError("No valid structure files could be parsed.")

    if use_default_feature_selection:
        feature_frame = FeatureExtract().get_features(structures)
    else:
        feature_frame = FeatureExtract().get_features(
            structures,
            select_col=select_columns or PAPER_FEATURE_COLUMNS,
        )
    if len(feature_frame) != len(sample_ids):
        raise RuntimeError(
            "Feature extraction returned fewer rows than parsed structures. "
            "Please inspect structures with missing elemental or structural properties."
        )

    result = pd.concat(
        [pd.DataFrame({id_column: sample_ids}), feature_frame.reset_index(drop=True)],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return FeatureExtractionResult(
        output_path=output_path,
        total_files=len(structure_paths),
        parsed_files=len(structures),
        skipped_files=skipped,
    )


def find_structure_files(
    structure_dir: Path,
    *,
    recursive: bool = False,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Find supported structure files in a directory."""

    if not structure_dir.exists():
        raise FileNotFoundError(f"Structure directory does not exist: {structure_dir}")
    if not structure_dir.is_dir():
        raise NotADirectoryError(f"Structure path is not a directory: {structure_dir}")

    normalized_extensions = tuple(extension.lower() for extension in extensions)
    iterator = structure_dir.rglob("*") if recursive else structure_dir.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file()
        and (
            path.suffix.lower() in normalized_extensions
            or path.name.upper() in DEFAULT_FILENAMES
        )
    ]
    return sorted(files, key=lambda path: str(path).lower())
