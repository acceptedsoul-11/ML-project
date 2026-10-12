import os
import glob
import pandas as pd


PROCESSED_ROOT = "data/processed"


def find_processed_files():
    pattern = os.path.join(
        PROCESSED_ROOT,
        "user_*",
        "*.csv"
    )

    return sorted(
        glob.glob(pattern)
    )


if __name__ == "__main__":

    print("\n==========================================")
    print("GEOLIFE COORDINATE QUALITY CHECK")
    print("==========================================")

    files = find_processed_files()

    print(
        "\nProcessed files found:",
        len(files)
    )

    invalid_latitude = []
    invalid_longitude = []

    total_invalid = 0

    for index, file_path in enumerate(files, start=1):

        df = pd.read_csv(
            file_path,
            usecols=[
                "user_id",
                "trajectory_id",
                "segment_id",
                "latitude",
                "longitude",
                "timestamp"
            ]
        )

        # Invalid latitude
        bad_lat = df[
            (df["latitude"] < -90) |
            (df["latitude"] > 90)
        ]

        # Invalid longitude
        bad_lon = df[
            (df["longitude"] < -180) |
            (df["longitude"] > 180)
        ]

        if not bad_lat.empty:
            invalid_latitude.append(
                bad_lat
            )

        if not bad_lon.empty:
            invalid_longitude.append(
                bad_lon
            )

        total_invalid += (
            len(bad_lat) +
            len(bad_lon)
        )

        if index % 1000 == 0:
            print(
                f"Checked {index}/{len(files)} files..."
            )

    print("\n==========================================")
    print("RESULTS")
    print("==========================================")

    if invalid_latitude:

        invalid_latitude = pd.concat(
            invalid_latitude,
            ignore_index=True
        )

        print(
            "\nInvalid latitude records:",
            len(invalid_latitude)
        )

        print(
            "\nLatitude range of invalid records:"
        )

        print(
            "Minimum:",
            invalid_latitude["latitude"].min()
        )

        print(
            "Maximum:",
            invalid_latitude["latitude"].max()
        )

        print("\nExamples:")

        print(
            invalid_latitude.head(10).to_string(
                index=False
            )
        )

    else:

        print(
            "\nInvalid latitude records: 0"
        )


    if invalid_longitude:

        invalid_longitude = pd.concat(
            invalid_longitude,
            ignore_index=True
        )

        print(
            "\nInvalid longitude records:",
            len(invalid_longitude)
        )

        print(
            "\nLongitude range of invalid records:"
        )

        print(
            "Minimum:",
            invalid_longitude["longitude"].min()
        )

        print(
            "Maximum:",
            invalid_longitude["longitude"].max()
        )

        print("\nExamples:")

        print(
            invalid_longitude.head(10).to_string(
                index=False
            )
        )

    else:

        print(
            "\nInvalid longitude records: 0"
        )


    print("\n==========================================")
    print(
        "Total invalid coordinate values:",
        total_invalid
    )
    print("==========================================")