import os
import glob
import pandas as pd


PROCESSED_ROOT = "data/processed"


def find_files():
    return sorted(
        glob.glob(
            os.path.join(
                PROCESSED_ROOT,
                "user_*",
                "*.csv"
            )
        )
    )


if __name__ == "__main__":

    print("\n==========================================")
    print("GEOGRAPHIC EXTREME AUDIT")
    print("==========================================")

    files = find_files()

    print(
        "\nProcessed files:",
        len(files)
    )

    extreme_records = []

    total_rows = 0

    for index, file_path in enumerate(files, start=1):

        df = pd.read_csv(
            file_path,
            usecols=[
                "user_id",
                "trajectory_id",
                "latitude",
                "longitude",
                "timestamp"
            ]
        )

        total_rows += len(df)

        # Look for geographically unusual values.
        unusual = df[
            (df["latitude"] < 10) |
            (df["latitude"] > 50) |
            (df["longitude"] < -170) |
            (df["longitude"] > 170)
        ].copy()

        if not unusual.empty:

            unusual["source_file"] = file_path

            extreme_records.append(
                unusual
            )

        if index % 1000 == 0:

            print(
                f"Checked {index}/{len(files)} files..."
            )

    print("\n==========================================")
    print("AUDIT RESULTS")
    print("==========================================")

    print(
        "Total GPS observations:",
        total_rows
    )

    if not extreme_records:

        print(
            "\nNo geographically unusual records found."
        )

    else:

        extreme_records = pd.concat(
            extreme_records,
            ignore_index=True
        )

        print(
            "\nUnusual geographic records:",
            len(extreme_records)
        )

        # ----------------------------------------------------
        # LATITUDE EXTREMES
        # ----------------------------------------------------

        print("\n==========================================")
        print("LOWEST LATITUDES")
        print("==========================================")

        print(
            extreme_records
            .sort_values("latitude")
            [
                [
                    "user_id",
                    "trajectory_id",
                    "latitude",
                    "longitude",
                    "timestamp"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        print("\n==========================================")
        print("HIGHEST LATITUDES")
        print("==========================================")

        print(
            extreme_records
            .sort_values(
                "latitude",
                ascending=False
            )
            [
                [
                    "user_id",
                    "trajectory_id",
                    "latitude",
                    "longitude",
                    "timestamp"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # LONGITUDE EXTREMES
        # ----------------------------------------------------

        print("\n==========================================")
        print("LOWEST LONGITUDES")
        print("==========================================")

        print(
            extreme_records
            .sort_values("longitude")
            [
                [
                    "user_id",
                    "trajectory_id",
                    "latitude",
                    "longitude",
                    "timestamp"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        print("\n==========================================")
        print("HIGHEST LONGITUDES")
        print("==========================================")

        print(
            extreme_records
            .sort_values(
                "longitude",
                ascending=False
            )
            [
                [
                    "user_id",
                    "trajectory_id",
                    "latitude",
                    "longitude",
                    "timestamp"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\n==========================================")
    print("AUDIT COMPLETE")
    print("==========================================")