import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_FILE = (
    "data/raw/Geolife Trajectories 1.3/"
    "Data/020/Trajectory/20110911000506.plt"
)

TARGET_DATE = "2011-09-11"
TARGET_TIME = "03:23:54"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("INSPECTING ORIGINAL GEOLIFE RECORD")
    print("==========================================")

    print("\nRaw file:")
    print(RAW_FILE)

    if not os.path.exists(RAW_FILE):

        print("\n[ERROR] Raw .plt file not found.")

        print(
            "\nMake sure you are running this from:"
        )

        print(
            "C:\\Users\\saira\\OneDrive\\Desktop\\ML-Projec_work"
        )

        raise SystemExit

    # --------------------------------------------------------
    # READ RAW FILE
    # --------------------------------------------------------

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
        RAW_FILE,
        skiprows=6,
        header=None,
        names=columns
    )

    # --------------------------------------------------------
    # CREATE TIMESTAMP
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"]
    )

    # --------------------------------------------------------
    # FIND TARGET RECORD
    # --------------------------------------------------------

    target = df[
        (df["date"] == TARGET_DATE) &
        (df["time"] == TARGET_TIME)
    ]

    print("\n==========================================")
    print("TARGET RECORD")
    print("==========================================")

    if target.empty:

        print(
            "\n[WARNING] Exact timestamp not found."
        )

    else:

        print(
            target[
                [
                    "latitude",
                    "longitude",
                    "altitude",
                    "date_days",
                    "date",
                    "time",
                    "timestamp"
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # SHOW RECORDS AROUND TARGET
    # --------------------------------------------------------

    if not target.empty:

        target_index = target.index[0]

        start = max(
            0,
            target_index - 5
        )

        end = min(
            len(df),
            target_index + 6
        )

        nearby = df.iloc[start:end]

        print("\n==========================================")
        print("NEARBY RAW RECORDS")
        print("==========================================")

        print(
            nearby[
                [
                    "latitude",
                    "longitude",
                    "altitude",
                    "date_days",
                    "date",
                    "time"
                ]
            ].to_string(index=True)
        )

    # --------------------------------------------------------
    # CHECK RAW LATITUDE VALIDITY
    # --------------------------------------------------------

    invalid_latitude = df[
        (df["latitude"] < -90) |
        (df["latitude"] > 90)
    ]

    print("\n==========================================")
    print("RAW FILE COORDINATE CHECK")
    print("==========================================")

    print(
        "Total observations:",
        len(df)
    )

    print(
        "Invalid latitude records:",
        len(invalid_latitude)
    )

    if not invalid_latitude.empty:

        print("\nInvalid latitude records:")

        print(
            invalid_latitude[
                [
                    "latitude",
                    "longitude",
                    "date",
                    "time"
                ]
            ].to_string(index=False)
        )

    print("\n==========================================")