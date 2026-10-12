import os
import pandas as pd


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "user_020/"
    "20110911000506.csv"
)

RAW_FILE = (
    "data/raw/Geolife Trajectories 1.3/"
    "Data/020/Trajectory/"
    "20110911000506.plt"
)


# ============================================================
# LOAD
# ============================================================

def load_processed_file():

    return pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("FIX INVALID GEOLIFE COORDINATE")
    print("==========================================")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(
            "\n[ERROR] Processed file not found:"
        )

        print(INPUT_FILE)

        raise SystemExit

    if not os.path.exists(RAW_FILE):

        print(
            "\n[ERROR] Raw file not found:"
        )

        print(RAW_FILE)

        raise SystemExit

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_processed_file()

    print(
        "\nOriginal observations:",
        len(df)
    )

    # --------------------------------------------------------
    # FIND INVALID COORDINATES
    # --------------------------------------------------------

    invalid = (
        (df["latitude"] < -90) |
        (df["latitude"] > 90) |
        (df["longitude"] < -180) |
        (df["longitude"] > 180)
    )

    invalid_count = invalid.sum()

    print(
        "Invalid coordinate records:",
        invalid_count
    )

    if invalid_count == 0:

        print(
            "\nNo invalid coordinates found."
        )

        raise SystemExit

    # --------------------------------------------------------
    # DISPLAY INVALID RECORD
    # --------------------------------------------------------

    print("\nInvalid record:")

    print(
        df.loc[
            invalid,
            [
                "user_id",
                "trajectory_id",
                "segment_id",
                "latitude",
                "longitude",
                "timestamp"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # REMOVE INVALID OBSERVATIONS
    # --------------------------------------------------------

    df_clean = df.loc[
        ~invalid
    ].copy()

    # --------------------------------------------------------
    # RECALCULATE TIME DIFFERENCE
    # --------------------------------------------------------

    df_clean["time_diff_seconds"] = (
        df_clean
        .groupby("segment_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # --------------------------------------------------------
    # RECALCULATE PREVIOUS COORDINATES
    # --------------------------------------------------------

    df_clean["previous_latitude"] = (
        df_clean
        .groupby("segment_id")["latitude"]
        .shift(1)
    )

    df_clean["previous_longitude"] = (
        df_clean
        .groupby("segment_id")["longitude"]
        .shift(1)
    )

    # --------------------------------------------------------
    # HAVERSINE DISTANCE
    # --------------------------------------------------------

    from math import radians, sin, cos, sqrt, atan2

    def haversine(
        lat1,
        lon1,
        lat2,
        lon2
    ):

        earth_radius = 6371000

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

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    df_clean["distance_meters"] = df_clean.apply(

        lambda row: haversine(
            row["previous_latitude"],
            row["previous_longitude"],
            row["latitude"],
            row["longitude"]
        )
        if pd.notna(row["previous_latitude"])
        else 0.0,

        axis=1
    )

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    df_clean["speed_mps"] = 0.0

    valid_time = (
        df_clean["time_diff_seconds"] > 0
    )

    df_clean.loc[
        valid_time,
        "speed_mps"
    ] = (
        df_clean.loc[
            valid_time,
            "distance_meters"
        ]
        /
        df_clean.loc[
            valid_time,
            "time_diff_seconds"
        ]
    )

    df_clean["speed_kmh"] = (
        df_clean["speed_mps"] * 3.6
    )

    # --------------------------------------------------------
    # REMOVE TEMPORARY COLUMNS
    # --------------------------------------------------------

    df_clean = df_clean[
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

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df_clean.to_csv(
        INPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print("\n==========================================")
    print("CLEANING COMPLETE")
    print("==========================================")

    print(
        "Original observations:",
        len(df)
    )

    print(
        "Removed invalid observations:",
        invalid_count
    )

    print(
        "Clean observations:",
        len(df_clean)
    )

    print(
        "Remaining invalid latitudes:",
        (
            (df_clean["latitude"] < -90) |
            (df_clean["latitude"] > 90)
        ).sum()
    )

    print(
        "Remaining invalid longitudes:",
        (
            (df_clean["longitude"] < -180) |
            (df_clean["longitude"] > 180)
        ).sum()
    )

    print(
        "\nUpdated file:",
        INPUT_FILE
    )

    print("==========================================")