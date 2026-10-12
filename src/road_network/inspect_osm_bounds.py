from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "region_analysis"
    / "trajectory_regions.csv"
)

TARGET_REGIONS = [
    "LAT_35_LON_115",
    "LAT_40_LON_115",
]

BUFFER_DEGREES = 0.05


# ============================================================
# Load region data
# ============================================================

def load_region_data():
    if not REGION_FILE.exists():
        raise FileNotFoundError(
            f"Region file not found:\n{REGION_FILE}"
        )

    df = pd.read_csv(REGION_FILE)

    print(f"Loaded region data: {len(df):,} rows")

    return df


# ============================================================
# Calculate combined bounds
# ============================================================

def calculate_bounds(df):

    selected = df[
        df["region_id"].isin(TARGET_REGIONS)
    ].copy()

    if selected.empty:
        raise ValueError(
            "None of the target regions were found."
        )

    print("\nSelected regions:")
    for region in TARGET_REGIONS:
        count = (selected["region_id"] == region).sum()
        print(f"  {region}: {count:,} trajectories")

    min_lat = selected["min_latitude"].min()
    max_lat = selected["max_latitude"].max()
    min_lon = selected["min_longitude"].min()
    max_lon = selected["max_longitude"].max()

    # Add a small geographic buffer
    min_lat -= BUFFER_DEGREES
    max_lat += BUFFER_DEGREES
    min_lon -= BUFFER_DEGREES
    max_lon += BUFFER_DEGREES

    return min_lat, max_lat, min_lon, max_lon


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("OSM ROAD NETWORK BOUNDARY INSPECTION")
    print("=" * 60)

    df = load_region_data()

    min_lat, max_lat, min_lon, max_lon = calculate_bounds(df)

    print("\nFinal bounding box:")
    print(f"  Minimum latitude : {min_lat:.6f}")
    print(f"  Maximum latitude : {max_lat:.6f}")
    print(f"  Minimum longitude: {min_lon:.6f}")
    print(f"  Maximum longitude: {max_lon:.6f}")

    print("\nOSMnx bbox format:")
    print(
        f"  (left, bottom, right, top) = "
        f"({min_lon:.6f}, {min_lat:.6f}, "
        f"{max_lon:.6f}, {max_lat:.6f})"
    )

    print("\nNo OSM data downloaded yet.")
    print("This step only inspects the requested area.")


if __name__ == "__main__":
    main()