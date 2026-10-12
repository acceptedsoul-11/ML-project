import os
import glob
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ============================================================
# 1. LOAD ONE GEOLIFE TRAJECTORY
# ============================================================

def load_trajectory(file_path):

    columns = [
        "latitude",
        "longitude",
        "unused",
        "altitude",
        "date_days",
        "date",
        "time"
    ]

    df = pd.read_csv(
        file_path,
        skiprows=6,
        header=None,
        names=columns
    )

    # Combine date and time
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"]
    )

    # Keep required columns
    df = df[
        [
            "latitude",
            "longitude",
            "altitude",
            "timestamp"
        ]
    ]

    return df


# ============================================================
# 2. SPLIT TRAJECTORY INTO CONTINUOUS SEGMENTS
# ============================================================

def split_trajectory(df, gap_threshold=60):

    df = df.copy()

    # Time difference from previous observation
    df["time_diff_seconds"] = (
        df["timestamp"].diff().dt.total_seconds()
    )

    # New segment whenever gap > 60 seconds
    df["segment_id"] = (
        df["time_diff_seconds"] > gap_threshold
    ).cumsum()

    return df


# ============================================================
# 3. HAVERSINE DISTANCE
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371000  # meters

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return earth_radius * c


# ============================================================
# 4. CALCULATE MOVEMENT
# ============================================================

def calculate_movement(df):

    df = df.copy()

    # Previous point within the same segment
    df["previous_latitude"] = (
        df.groupby("segment_id")["latitude"]
        .shift(1)
    )

    df["previous_longitude"] = (
        df.groupby("segment_id")["longitude"]
        .shift(1)
    )

    # Calculate distance
    df["distance_meters"] = df.apply(
        lambda row: calculate_distance(
            row["previous_latitude"],
            row["previous_longitude"],
            row["latitude"],
            row["longitude"]
        )
        if pd.notna(row["previous_latitude"])
        else 0.0,
        axis=1
    )

    # Calculate speed
    df["speed_mps"] = 0.0

    valid_time = (
        df["time_diff_seconds"] > 0
    )

    df.loc[valid_time, "speed_mps"] = (
        df.loc[valid_time, "distance_meters"]
        / df.loc[valid_time, "time_diff_seconds"]
    )

    # Convert m/s to km/h
    df["speed_kmh"] = (
        df["speed_mps"] * 3.6
    )

    return df


# ============================================================
# 5. ADD METADATA
# ============================================================

def add_metadata(df, user_id, trajectory_id):

    df = df.copy()

    df.insert(
        0,
        "user_id",
        user_id
    )

    df.insert(
        1,
        "trajectory_id",
        trajectory_id
    )

    return df


# ============================================================
# 6. SAVE PROCESSED TRAJECTORY
# ============================================================

def save_processed_trajectory(df, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df = df[
        [
            "user_id",
            "trajectory_id",
            "segment_id",
            "latitude",
            "longitude",
            "altitude",
            "timestamp",
            "time_diff_seconds",
            "distance_meters",
            "speed_kmh"
        ]
    ]

    df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # DATASET ROOT
    # --------------------------------------------------------

    data_root = (
        "data/raw/Geolife Trajectories 1.3/Data"
    )

    # --------------------------------------------------------
    # OUTPUT ROOT
    # --------------------------------------------------------

    output_root = (
        "data/processed"
    )

    # --------------------------------------------------------
    # FIND ALL USER DIRECTORIES
    # --------------------------------------------------------

    user_directories = sorted(
        [
            directory
            for directory in glob.glob(
                os.path.join(
                    data_root,
                    "*"
                )
            )
            if os.path.isdir(directory)
        ]
    )

    print("\n==========================================")
    print("GEOLIFE FULL DATASET PREPROCESSING")
    print("==========================================")

    print(
        "\nUser directories found:",
        len(user_directories)
    )

    print("\nProcessing all users...\n")

    # --------------------------------------------------------
    # GLOBAL COUNTERS
    # --------------------------------------------------------

    total_users = 0
    total_files_found = 0
    total_files_processed = 0
    total_errors = 0
    total_observations = 0
    total_segments = 0

    # ========================================================
    # PROCESS EVERY USER
    # ========================================================

    for user_directory in user_directories:

        # User ID
        user_id = os.path.basename(
            user_directory
        )

        # Trajectory directory
        trajectory_directory = os.path.join(
            user_directory,
            "Trajectory"
        )

        # Check whether Trajectory exists
        if not os.path.isdir(
            trajectory_directory
        ):
            print(
                f"[SKIP] User {user_id}: "
                "Trajectory folder not found"
            )
            continue

        # Find all .plt files
        trajectory_files = sorted(
            glob.glob(
                os.path.join(
                    trajectory_directory,
                    "*.plt"
                )
            )
        )

        total_users += 1

        total_files_found += len(
            trajectory_files
        )

        print(
            f"\nUSER {user_id}"
        )

        print(
            f"Trajectory files: "
            f"{len(trajectory_files)}"
        )

        # ----------------------------------------------------
        # OUTPUT DIRECTORY FOR THIS USER
        # ----------------------------------------------------

        user_output_directory = os.path.join(
            output_root,
            f"user_{user_id}"
        )

        # ----------------------------------------------------
        # PROCESS EACH TRAJECTORY
        # ----------------------------------------------------

        for file_path in trajectory_files:

            filename = os.path.basename(
                file_path
            )

            trajectory_id = os.path.splitext(
                filename
            )[0]

            try:

                # Load
                df = load_trajectory(
                    file_path
                )

                # Segment
                df = split_trajectory(
                    df,
                    gap_threshold=60
                )

                # Movement
                df = calculate_movement(
                    df
                )

                # Metadata
                df = add_metadata(
                    df,
                    user_id,
                    trajectory_id
                )

                # Output path
                output_path = os.path.join(
                    user_output_directory,
                    trajectory_id + ".csv"
                )

                # Save
                save_processed_trajectory(
                    df,
                    output_path
                )

                # Update counters
                total_files_processed += 1

                total_observations += len(df)

                total_segments += (
                    df["segment_id"].nunique()
                )

                print(
                    f"  [OK] {trajectory_id} | "
                    f"observations={len(df)} | "
                    f"segments="
                    f"{df['segment_id'].nunique()}"
                )

            except Exception as error:

                total_errors += 1

                print(
                    f"  [ERROR] {trajectory_id} | "
                    f"{error}"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n==========================================")
    print("FULL DATASET PROCESSING COMPLETE")
    print("==========================================")

    print(
        "Users processed:",
        total_users
    )

    print(
        "Trajectory files found:",
        total_files_found
    )

    print(
        "Trajectory files processed:",
        total_files_processed
    )

    print(
        "Files with errors:",
        total_errors
    )

    print(
        "Total GPS observations:",
        total_observations
    )

    print(
        "Total trajectory segments:",
        total_segments
    )

    print(
        "Output directory:",
        output_root
    )

    print("==========================================")