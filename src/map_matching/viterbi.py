
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "map_matching"
    / "20081023025304_emission_probabilities.csv"
)

TRANSITIONS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "map_matching"
    / "20081023025304_transition_probabilities.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "map_matching"
    / "20081023025304_matched_route.csv"
)

GROUP_COLUMNS = ["user_id", "trajectory_id", "segment_id"]


def viterbi_decode(
    emission_scores: list[np.ndarray],
    transition_scores: list[np.ndarray],
) -> tuple[list[int] | None, list[np.ndarray], float]:
    """Decode one connected candidate run using the project's log-score model."""
    if not emission_scores:
        return None, [], -np.inf
    if len(transition_scores) != len(emission_scores) - 1:
        raise ValueError("A transition matrix is required between each pair of observations.")
    emissions = [np.asarray(scores, dtype=float) for scores in emission_scores]
    scores = emissions[0].copy()
    score_history = [scores.copy()]
    backpointers = [np.full(len(scores), -1, dtype=int)]
    for index, transitions in enumerate(transition_scores, start=1):
        transitions = np.asarray(transitions, dtype=float)
        if transitions.shape != (len(scores), len(emissions[index])):
            raise ValueError("Transition matrix dimensions do not match candidate counts.")
        combined = scores[:, None] + transitions
        previous = np.argmax(combined, axis=0)
        best_previous = combined[previous, np.arange(combined.shape[1])]
        scores = best_previous + emissions[index]
        score_history.append(scores.copy())
        backpointers.append(previous)
    if not np.isfinite(scores).any():
        return None, score_history, -np.inf
    best_final = float(np.max(scores))
    path = [int(np.argmax(scores))]
    for index in range(len(emissions) - 1, 0, -1):
        path.append(int(backpointers[index][path[-1]]))
    path.reverse()
    return path, score_history, best_final


def main():
    print("Loading emission and transition data...")

    candidates = pd.read_csv(CANDIDATES_FILE)
    transitions = pd.read_csv(TRANSITIONS_FILE)

    required_candidates = {
        "user_id", "trajectory_id", "segment_id",
        "point_number", "timestamp",
        "gps_latitude", "gps_longitude",
        "u", "v", "key", "distance_meters",
        "emission_log_score",
    }

    required_transitions = {
        "user_id", "trajectory_id", "segment_id",
        "previous_point_number", "current_point_number",
        "previous_candidate_id", "current_candidate_id",
        "transition_log_score",
    }

    missing = required_candidates - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    missing = required_transitions - set(transitions.columns)
    if missing:
        raise ValueError(
            f"Missing transition columns: {sorted(missing)}"
        )

    if candidates.empty:
        raise ValueError("The candidate CSV is empty.")

    # IDs are the original row positions used by the transition script.
    # Do not sort or filter candidates before assigning these IDs.
    candidates = candidates.reset_index(drop=True)
    candidates["candidate_id"] = np.arange(len(candidates))

    candidates["timestamp"] = pd.to_datetime(
        candidates["timestamp"], errors="raise"
    )

    candidates["emission_log_score"] = pd.to_numeric(
        candidates["emission_log_score"], errors="raise"
    )

    for column in [
        "previous_candidate_id",
        "current_candidate_id",
        "previous_point_number",
        "current_point_number",
        "user_id",
        "trajectory_id",
        "segment_id",
    ]:
        transitions[column] = pd.to_numeric(
            transitions[column], errors="raise"
        ).astype(int)

    transitions["transition_log_score"] = pd.to_numeric(
        transitions["transition_log_score"],
        errors="coerce",
    )

    # Map each observation pair to its candidate transition scores.
    transition_lookup = {}

    transition_group_columns = (
        GROUP_COLUMNS
        + ["previous_point_number", "current_point_number"]
    )

    for key, group in transitions.groupby(
        transition_group_columns,
        sort=False,
        dropna=False,
    ):
        transition_lookup[key] = {
            (
                int(row["previous_candidate_id"]),
                int(row["current_candidate_id"]),
            ): float(row["transition_log_score"])
            for _, row in group.iterrows()
            if np.isfinite(row["transition_log_score"])
        }

    matched_parts = []
    skipped_segments = 0

    print("Running Viterbi decoding...")

    for segment_key, segment in candidates.groupby(
        GROUP_COLUMNS,
        sort=False,
        dropna=False,
    ):
        segment = segment.sort_values(
            ["point_number", "timestamp"],
            kind="stable",
        )

        observations = [
            group.copy()
            for _, group in segment.groupby(
                "point_number", sort=True
            )
        ]

        if not observations:
            continue

        emission_scores = [
            observation["emission_log_score"].to_numpy(dtype=float)
            for observation in observations
        ]
        transition_scores = []

        # Build a score matrix between each pair of observations.
        for t in range(1, len(observations)):
            previous_observation = observations[t - 1]
            current_observation = observations[t]

            previous_point = int(
                previous_observation["point_number"].iloc[0]
            )
            current_point = int(
                current_observation["point_number"].iloc[0]
            )

            transition_key = (
                int(segment_key[0]),
                int(segment_key[1]),
                int(segment_key[2]),
                previous_point,
                current_point,
            )

            pair_scores = transition_lookup.get(
                transition_key, {}
            )
            matrix = np.full(
                (len(previous_observation), len(current_observation)),
                -np.inf,
                dtype=float,
            )
            for previous_index, previous in enumerate(
                previous_observation["candidate_id"].to_numpy(dtype=int)
            ):
                for current_index, current in enumerate(
                    current_observation["candidate_id"].to_numpy(dtype=int)
                ):
                    matrix[previous_index, current_index] = pair_scores.get(
                        (int(previous), int(current)), -np.inf
                    )
            transition_scores.append(matrix)

        path, score_history, best_final_score = viterbi_decode(
            emission_scores, transition_scores
        )
        if path is None:
            print(
                f"Warning: no complete path for segment "
                f"{segment_key}; skipping it."
            )
            skipped_segments += 1
            continue

        # Save one selected road candidate per GPS observation.
        for t, (observation, candidate_index) in enumerate(
            zip(observations, path)
        ):
            output_row = observation.iloc[candidate_index].to_dict()
            output_row["viterbi_cumulative_score"] = (
                score_history[t][candidate_index]
            )
            output_row["viterbi_best_path_score"] = best_final_score

            matched_parts.append(output_row)

    if not matched_parts:
        raise RuntimeError(
            "No matched route was produced. Check transition connectivity."
        )

    result = pd.DataFrame(matched_parts)

    result = result.sort_values(
        GROUP_COLUMNS + ["point_number"],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nViterbi decoding completed successfully.")
    print(f"Matched observations: {len(result)}")
    print(f"Skipped segments: {skipped_segments}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nFirst 10 matched observations:")

    print(
        result[
            [
                "point_number",
                "timestamp",
                "gps_latitude",
                "gps_longitude",
                "u",
                "v",
                "distance_meters",
                "emission_probability",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
