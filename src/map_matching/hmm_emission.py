
from pathlib import Path

import numpy as np
import pandas as pd


# Project root: ML-Projec_work
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "map_matching"
    / "20081023025304_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "map_matching"
    / "20081023025304_emission_probabilities.csv"
)

# Initial assumed GPS measurement error, in metres.
GPS_SIGMA_METERS = 20.0


def calculate_emission_probabilities(
    candidates: pd.DataFrame,
    gps_sigma_meters: float = GPS_SIGMA_METERS,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Score candidate-to-GPS distances with the project's Gaussian model."""
    if not np.isfinite(gps_sigma_meters) or gps_sigma_meters <= 0:
        raise ValueError("GPS sigma must be a positive finite number.")
    df = candidates.copy()
    if df.empty:
        raise ValueError("The candidate table contains no records.")
    if "distance_meters" not in df.columns:
        raise ValueError("The candidate table must contain 'distance_meters'.")
    if group_columns is None:
        possible = [
            "user_id", "trajectory_id", "segment_id", "timestamp",
            "latitude", "longitude", "point_number",
        ]
        group_columns = [column for column in possible if column in df.columns]
    missing_groups = set(group_columns) - set(df.columns)
    if not group_columns or missing_groups:
        raise ValueError("Candidate observation group columns are missing or empty.")
    df["distance_meters"] = pd.to_numeric(df["distance_meters"], errors="coerce")
    if df.distance_meters.isna().any() or (df.distance_meters < 0).any():
        raise ValueError("Candidate distances must be valid, non-negative numbers.")
    df["emission_log_score"] = -(df.distance_meters.to_numpy(dtype=float) ** 2) / (
        2 * gps_sigma_meters**2
    )
    df["emission_probability"] = 0.0
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for _, indices in df.groupby(group_key, sort=False, dropna=False).groups.items():
        log_scores = df.loc[indices, "emission_log_score"].to_numpy(dtype=float)
        weights = np.exp(log_scores - np.max(log_scores))
        df.loc[indices, "emission_probability"] = weights / weights.sum()
    return df


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {INPUT_FILE}"
        )

    raw = pd.read_csv(INPUT_FILE)
    possible_group_columns = [
        "user_id", "trajectory_id", "segment_id", "timestamp",
        "latitude", "longitude", "point_number",
    ]
    group_columns = [column for column in possible_group_columns if column in raw.columns]
    df = calculate_emission_probabilities(raw, GPS_SIGMA_METERS, group_columns)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    # Report a concise summary.
    number_of_observations = (
        df.groupby(group_columns, dropna=False)
        .ngroups
    )

    print("HMM emission model completed successfully.")
    print(f"GPS observations: {number_of_observations}")
    print(f"Candidate records: {len(df)}")
    print(f"Assumed GPS sigma: {GPS_SIGMA_METERS} metres")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nSample emission probabilities:")
    display_columns = [
        column
        for column in [
            "timestamp",
            "latitude",
            "longitude",
            "distance_meters",
            "emission_probability",
        ]
        if column in df.columns
    ]

    print(
        df[display_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
