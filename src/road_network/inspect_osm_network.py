from pathlib import Path

import osmnx as ox
import networkx as nx


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "road_network"
    / "beijing_drive.graphml"
)


# ============================================================
# Load graph
# ============================================================

def load_graph():

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"OSM graph not found:\n{GRAPH_FILE}"
        )

    print("Loading OSM road network...")

    G = ox.load_graphml(GRAPH_FILE)

    return G


# ============================================================
# Inspect graph
# ============================================================

def inspect_graph(G):

    print("\n" + "=" * 60)
    print("OSM ROAD NETWORK INSPECTION")
    print("=" * 60)

    # Basic graph information
    print("\nGraph information:")
    print(f"  Graph type       : {type(G).__name__}")
    print(f"  Nodes            : {len(G.nodes):,}")
    print(f"  Edges            : {len(G.edges):,}")

    # Directed graph check
    print(f"  Directed graph   : {G.is_directed()}")

    # CRS
    print(f"  CRS              : {G.graph.get('crs')}")

    # Network bounds
    nodes = list(G.nodes(data=True))

    if nodes:

        x_values = [
            float(data["x"])
            for _, data in nodes
            if "x" in data
        ]

        y_values = [
            float(data["y"])
            for _, data in nodes
            if "y" in data
        ]

        print("\nNode coordinate bounds:")

        print(
            f"  Longitude min    : {min(x_values):.6f}"
        )

        print(
            f"  Longitude max    : {max(x_values):.6f}"
        )

        print(
            f"  Latitude min     : {min(y_values):.6f}"
        )

        print(
            f"  Latitude max     : {max(y_values):.6f}"
        )

    # Connected components
    components = list(
        nx.weakly_connected_components(G)
    )

    print("\nConnectivity:")
    print(
        f"  Weak components  : {len(components):,}"
    )

    if components:

        largest_component = max(
            components,
            key=len
        )

        print(
            f"  Largest component: "
            f"{len(largest_component):,} nodes"
        )

    # Edge attributes
    if len(G.edges) > 0:

        first_edge = next(
            iter(G.edges(data=True))
        )

        _, _, edge_data = first_edge

        print("\nExample edge attributes:")

        for key, value in edge_data.items():

            # Avoid printing extremely large geometry objects
            if key == "geometry":
                print(
                    f"  {key:<15}: "
                    f"<geometry object>"
                )
            else:
                print(
                    f"  {key:<15}: {value}"
                )

    print("\n" + "=" * 60)
    print("OSM GRAPH INSPECTION COMPLETE")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    G = load_graph()

    inspect_graph(G)


if __name__ == "__main__":
    main()