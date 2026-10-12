"""Command-line runner for reproducible buffer/delay experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.delay_accuracy import (
    MODEL_FULL,
    MODEL_NEAREST,
    MODEL_SLIDING,
    run_delay_experiment,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="One or more processed GPS trajectory CSV files.")
    parser.add_argument("--output-dir", type=Path, help="Output run folder. Defaults to results/experiments/<UTC timestamp>.")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--buffer-sizes", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 10])
    parser.add_argument("--baselines", nargs="+", choices=[MODEL_NEAREST, MODEL_FULL, MODEL_SLIDING], default=[MODEL_NEAREST, MODEL_FULL, MODEL_SLIDING])
    parser.add_argument("--ground-truth-column", help="Optional column with directed edge labels formatted as (u,v,key) or u|v|key.")
    parser.add_argument("--radius-meters", type=float, default=100)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--gps-sigma-meters", type=float, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_frames = []
    for filename in args.input:
        path = (ROOT / filename).resolve() if not Path(filename).is_absolute() else Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory CSV not found: {path}")
        input_frames.append((str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), pd.read_csv(path)))
    outcome = run_delay_experiment(
        input_frames,
        window_sizes=args.window_sizes,
        buffer_sizes=args.buffer_sizes,
        baselines=args.baselines,
        ground_truth_column=args.ground_truth_column,
        radius_meters=args.radius_meters,
        max_candidates=args.max_candidates,
        gps_sigma_meters=args.gps_sigma_meters,
        output_dir=args.output_dir,
    )
    print(f"Experiment results: {outcome['output_dir'] / 'experiment_results.csv'}")
    print(f"Plots: {', '.join(outcome['plots'])}")
    columns = ["trajectory_id", "model", "window_size", "buffer_size", "coverage_percent", "ground_truth_accuracy_percent", "mean_prediction_delay_observations", "runtime_seconds"]
    print(outcome["results"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
