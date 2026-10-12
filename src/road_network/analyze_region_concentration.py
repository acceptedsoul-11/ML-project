import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "region_analysis/"
    "region_summary.csv"
)


if __name__ == "__main__":

    print("\n==========================================")
    print("REGION CONCENTRATION ANALYSIS")
    print("==========================================")

    df = pd.read_csv(INPUT_FILE)

    # Sort by number of observations
    df = df.sort_values(
        "observations",
        ascending=False
    ).reset_index(drop=True)

    total_observations = df["observations"].sum()

    total_trajectories = df["trajectories"].sum()

    print(
        "\nTotal geographic regions:",
        len(df)
    )

    print(
        "Total trajectories:",
        total_trajectories
    )

    print(
        "Total observations:",
        total_observations
    )

    # --------------------------------------------------------
    # ADD PERCENTAGES
    # --------------------------------------------------------

    df["observation_percentage"] = (
        df["observations"]
        / total_observations
        * 100
    )

    df["trajectory_percentage"] = (
        df["trajectories"]
        / total_trajectories
        * 100
    )

    # --------------------------------------------------------
    # CUMULATIVE OBSERVATION COVERAGE
    # --------------------------------------------------------

    df["cumulative_observation_percentage"] = (
        df["observation_percentage"].cumsum()
    )

    # --------------------------------------------------------
    # TOP 20 REGIONS
    # --------------------------------------------------------

    print("\n==========================================")
    print("TOP 20 REGIONS BY GPS OBSERVATIONS")
    print("==========================================")

    print(
        df.head(20)[
            [
                "region_id",
                "trajectories",
                "users",
                "observations",
                "observation_percentage",
                "cumulative_observation_percentage",
                "mean_latitude",
                "mean_longitude"
            ]
        ].to_string(
            index=False,
            formatters={
                "observation_percentage": "{:.2f}%".format,
                "cumulative_observation_percentage":
                    "{:.2f}%".format,
                "mean_latitude": "{:.4f}".format,
                "mean_longitude": "{:.4f}".format
            }
        )
    )

    # --------------------------------------------------------
    # TOP 5 COVERAGE
    # --------------------------------------------------------

    top5 = df.head(5)

    top5_percentage = (
        top5["observations"].sum()
        / total_observations
        * 100
    )

    print("\n==========================================")
    print("TOP 5 COVERAGE")
    print("==========================================")

    print(
        f"Top 5 regions contain "
        f"{top5_percentage:.2f}% "
        f"of all GPS observations."
    )

    # --------------------------------------------------------
    # TOP 10 COVERAGE
    # --------------------------------------------------------

    top10 = df.head(10)

    top10_percentage = (
        top10["observations"].sum()
        / total_observations
        * 100
    )

    print(
        f"Top 10 regions contain "
        f"{top10_percentage:.2f}% "
        f"of all GPS observations."
    )

    # --------------------------------------------------------
    # 80% COVERAGE
    # --------------------------------------------------------

    regions_for_80 = (
        df[
            df["cumulative_observation_percentage"]
            <= 80
        ]
    )

    # Include the region that crosses 80%
    if len(regions_for_80) < len(df):

        regions_for_80 = df.head(
            len(regions_for_80) + 1
        )

    print("\n==========================================")
    print("80% DATA COVERAGE")
    print("==========================================")

    print(
        "Regions needed to cover approximately "
        "80% of GPS observations:",
        len(regions_for_80)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        "data/processed/"
        "region_analysis/"
        "region_concentration.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\n==========================================")
    print("ANALYSIS COMPLETE")
    print("==========================================")

    print(
        "Saved:",
        output_file
    )

    print("==========================================")