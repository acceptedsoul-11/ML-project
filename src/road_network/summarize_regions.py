import os
import glob
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_ROOT = "data/processed"

# Size of geographic grid cell.
# 5 degrees gives us a broad first-level regional picture.
GRID_SIZE = 5


# ============================================================
# FIND PROCESSED FILES
# ============================================================

def find_files():

    pattern = os.path.join(
        PROCESSED_ROOT,
        "user_*",
        "*.csv"
    )

    return sorted(
        glob.glob(pattern)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("GEOLIFE GEOGRAPHIC REGION SUMMARY")
    print("==========================================")

    files = find_files()

    print(
        "\nProcessed trajectory files:",
        len(files)
    )

    trajectory_records = []

    # --------------------------------------------------------
    # PROCESS EACH TRAJECTORY
    # --------------------------------------------------------

    for index, file_path in enumerate(
        files,
        start=1
    ):

        df = pd.read_csv(
            file_path,
            usecols=[
                "user_id",
                "trajectory_id",
                "latitude",
                "longitude"
            ]
        )

        if df.empty:
            continue

        # ----------------------------------------------------
        # TRAJECTORY CENTROID
        # ----------------------------------------------------

        mean_latitude = df["latitude"].mean()
        mean_longitude = df["longitude"].mean()

        min_latitude = df["latitude"].min()
        max_latitude = df["latitude"].max()

        min_longitude = df["longitude"].min()
        max_longitude = df["longitude"].max()

        observation_count = len(df)

        # ----------------------------------------------------
        # GRID REGION
        # ----------------------------------------------------

        region_lat = (
            int(mean_latitude // GRID_SIZE)
            * GRID_SIZE
        )

        region_lon = (
            int(mean_longitude // GRID_SIZE)
            * GRID_SIZE
        )

        region_id = (
            f"LAT_{region_lat}_"
            f"LON_{region_lon}"
        )

        trajectory_records.append(
            {
                "user_id": df["user_id"].iloc[0],
                "trajectory_id": df["trajectory_id"].iloc[0],
                "mean_latitude": mean_latitude,
                "mean_longitude": mean_longitude,
                "min_latitude": min_latitude,
                "max_latitude": max_latitude,
                "min_longitude": min_longitude,
                "max_longitude": max_longitude,
                "observation_count": observation_count,
                "region_id": region_id
            }
        )

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index}/{len(files)} trajectories..."
            )

    # --------------------------------------------------------
    # CREATE SUMMARY DATAFRAME
    # --------------------------------------------------------

    trajectories = pd.DataFrame(
        trajectory_records
    )

    print("\n==========================================")
    print("TRAJECTORY SUMMARY")
    print("==========================================")

    print(
        "Trajectories summarized:",
        len(trajectories)
    )

    # --------------------------------------------------------
    # REGION SUMMARY
    # --------------------------------------------------------

    regions = (
        trajectories
        .groupby("region_id")
        .agg(
            trajectories=(
                "trajectory_id",
                "count"
            ),
            users=(
                "user_id",
                "nunique"
            ),
            observations=(
                "observation_count",
                "sum"
            ),
            mean_latitude=(
                "mean_latitude",
                "mean"
            ),
            mean_longitude=(
                "mean_longitude",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "observations",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n==========================================")
    print("GEOGRAPHIC REGIONS")
    print("==========================================")

    print(
        regions.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    output_directory = (
        "data/processed/region_analysis"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    trajectory_output = os.path.join(
        output_directory,
        "trajectory_regions.csv"
    )

    region_output = os.path.join(
        output_directory,
        "region_summary.csv"
    )

    trajectories.to_csv(
        trajectory_output,
        index=False
    )

    regions.to_csv(
        region_output,
        index=False
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n==========================================")
    print("REGION ANALYSIS COMPLETE")
    print("==========================================")

    print(
        "Unique geographic regions:",
        len(regions)
    )

    print(
        "\nTrajectory-level file:"
    )

    print(
        trajectory_output
    )

    print(
        "\nRegion summary file:"
    )

    print(
        region_output
    )

    print("==========================================")