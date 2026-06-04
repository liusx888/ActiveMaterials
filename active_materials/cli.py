"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning tools for materials screening.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one active-learning selection round.")
    run_parser.add_argument("--config", type=Path, default=Path("configs/example.yaml"))
    run_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    extract_parser = subparsers.add_parser(
        "extract-features",
        help="Extract composition and structure features from POSCAR/VASP/CIF files.",
    )
    extract_parser.add_argument("--structure-dir", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--recursive", action="store_true")
    extract_parser.add_argument("--id-column", default="formula")
    extract_parser.add_argument(
        "--all-features",
        action="store_true",
        help="Keep FeatureExtract's default behavior instead of the paper's fixed feature columns.",
    )
    extract_parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".cif", ".vasp", ".poscar"],
        help="File extensions to parse. POSCAR and CONTCAR names are always accepted.",
    )
    extract_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    reduce_parser = subparsers.add_parser(
        "select-features",
        help="Select or reduce feature columns from a CSV file.",
    )
    reduce_parser.add_argument("--input", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.add_argument("--n-features", type=int, required=True)
    reduce_parser.add_argument(
        "--method",
        default="variance",
        choices=["variance", "correlation", "random_forest", "mutual_info", "pca"],
    )
    reduce_parser.add_argument("--id-column", default="0")
    reduce_parser.add_argument("--target-column", default=None)
    reduce_parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=[],
        help="Extra columns to exclude before feature selection, for example B.",
    )
    reduce_parser.add_argument("--selected-features-output", type=Path, default=None)
    reduce_parser.add_argument("--random-seed", type=int, default=1007)
    reduce_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Shortcut for the default run command.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Shortcut log level for the default run command.",
    )
    args = parser.parse_args()

    log_level = args.log_level or "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.command == "extract-features":
        from active_materials.extractor import extract_features_from_directory

        result = extract_features_from_directory(
            structure_dir=args.structure_dir,
            output_path=args.output,
            recursive=args.recursive,
            extensions=tuple(args.extensions),
            id_column=args.id_column,
            use_default_feature_selection=args.all_features,
        )
        print(f"Feature file: {result.output_path}")
        print(f"Total files: {result.total_files}")
        print(f"Parsed files: {result.parsed_files}")
        print(f"Skipped files: {result.skipped_files}")
        return

    if args.command == "select-features":
        from active_materials.selection import reduce_features

        result = reduce_features(
            input_csv=args.input,
            output_csv=args.output,
            n_features=args.n_features,
            method=args.method,
            id_column=_parse_column_arg(args.id_column),
            target_column=_parse_column_arg(args.target_column),
            exclude_columns=[_parse_column_arg(value) for value in args.exclude_columns],
            random_seed=args.random_seed,
            selected_features_output=args.selected_features_output,
        )
        print(f"Method: {result.method}")
        print(f"Input feature count: {result.input_feature_count}")
        print(f"Output feature count: {result.output_feature_count}")
        print(f"Reduced feature file: {result.output_path}")
        print(f"Selected feature list: {result.selected_features_path}")
        return

    from active_materials.config import PipelineConfig
    from active_materials.pipeline import ActiveLearningPipeline

    config_path = args.config or Path("configs/example.yaml")
    ActiveLearningPipeline(PipelineConfig.from_yaml(config_path)).run()


def _parse_column_arg(value: str | None) -> str | int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
