import os
import glob
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_ROOT = "data/processed"

# Safety buffer around GPS extent, in degrees.
# Approximately 1 degree latitude ~= 111 km.
BUFFER_DEGREES = 0.01


# ============================================================
# FIND ALL PROCESSED CSV FILES
# ============================================================

def find_processed_files():

    pattern = os.path.join(
        PROCESSED_ROOT,
        "user_*",
        "*.csv"
    )

    return sorted(
        glob.glob(pattern)
    )


# ============================================================
# CALCULATE GPS BOUNDS
# ============================================================

def calculate_bounds(files):

    min_latitude = float("inf")
    max_latitude = float("-inf")

    min_longitude = float("inf")
    max_longitude = float("-inf")

    total_rows = 0

    for index, file_path in enumerate(files, start=1):

        df = pd.read_csv(
            file_path,
            usecols=[
                "latitude",
                "longitude"
            ]
        )

        if df.empty:
            continue

        min_latitude = min(
            min_latitude,
            df["latitude"].min()
        )

        max_latitude = max(
            max_latitude,
            df["latitude"].max()
        )

        min_longitude = min(
            min_longitude,
            df["longitude"].min()
        )

        max_longitude = max(
            max_longitude,
            df["longitude"].max()
        )

        total_rows += len(df)

        if index % 500 == 0:
            print(
                f"Processed {index}/{len(files)} files..."
            )

    return (
        min_latitude,
        max_latitude,
        min_longitude,
        max_longitude,
        total_rows
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("GEOLIFE GPS GEOGRAPHIC EXTENT")
    print("==========================================")

    files = find_processed_files()

    print(
        "\nProcessed trajectory files found:",
        len(files)
    )

    if not files:
        print(
            "\n[ERROR] No processed CSV files found."
        )

        raise SystemExit

    print(
        "\nCalculating geographic bounds..."
    )

    (
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        total_rows
    ) = calculate_bounds(files)

    # --------------------------------------------------------
    # ADD BUFFER
    # --------------------------------------------------------

    buffered_min_lat = (
        min_lat - BUFFER_DEGREES
    )

    buffered_max_lat = (
        max_lat + BUFFER_DEGREES
    )

    buffered_min_lon = (
        min_lon - BUFFER_DEGREES
    )

    buffered_max_lon = (
        max_lon + BUFFER_DEGREES
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n==========================================")
    print("GPS EXTENT")
    print("==========================================")

    print(
        f"Minimum latitude : {min_lat:.6f}"
    )

    print(
        f"Maximum latitude : {max_lat:.6f}"
    )

    print(
        f"Minimum longitude: {min_lon:.6f}"
    )

    print(
        f"Maximum longitude: {max_lon:.6f}"
    )

    print(
        "\nTotal GPS observations checked:",
        total_rows
    )

    print("\n==========================================")
    print("BUFFERED EXTENT")
    print("==========================================")

    print(
        f"Minimum latitude : {buffered_min_lat:.6f}"
    )

    print(
        f"Maximum latitude : {buffered_max_lat:.6f}"
    )

    print(
        f"Minimum longitude: {buffered_min_lon:.6f}"
    )

    print(
        f"Maximum longitude: {buffered_max_lon:.6f}"
    )

    print("\n==========================================")
    print("NEXT STEP")
    print("==========================================")

    print(
        "Use the buffered extent to obtain the"
    )

    print(
        "OpenStreetMap road network."
    )

    print("==========================================")