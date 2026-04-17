"""
Preprocessing Script for Fraud Detection
-----------------------------------------
Reads edges.csv (with string node IDs like C..., M...), builds a NetworkX
graph, computes structural features (degree, PageRank, clustering coefficient),
merges with raw transactional data, maps all string IDs to integer indices,
and outputs numeric versions of features.csv, edges.csv, and labels.csv
for the GNN pipeline.

Usage:
    python preprocess.py

Input files  (in ../data/):
    - edges.csv        : CSV edge list with string IDs (src, dst)
    - raw_features.csv : raw transactional data per node
                         columns: node, tx_count, total_amount, avg_amount,
                                  max_amount, std_amount, tx_frequency
    - labels.csv       : node labels with string IDs (node, label)

Output files (in ../data/):
    - features.csv     : merged structural + transactional features (numeric node IDs)
    - edges.csv        : numeric edge list for load_data.py (overwrites input)
    - labels.csv       : numeric labels for load_data.py (overwrites input)
    - node_mapping.csv : string ID -> numeric ID lookup table
"""

import os
import pandas as pd
import networkx as nx

# -- Paths -----------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EDGES_CSV = os.path.join(DATA_DIR, "edges.csv")
RAW_FEATURES = os.path.join(DATA_DIR, "raw_features.csv")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
OUTPUT_FEATURES = os.path.join(DATA_DIR, "features.csv")
OUTPUT_EDGES = os.path.join(DATA_DIR, "edges.csv")
OUTPUT_LABELS = os.path.join(DATA_DIR, "labels.csv")
OUTPUT_MAPPING = os.path.join(DATA_DIR, "node_mapping.csv")


# --------------------------------------------------------------------------
# Step 1: Build node mapping (string ID -> integer)
# --------------------------------------------------------------------------
def build_node_mapping(edges_df, raw_df, labels_df):
    """Collect all unique node IDs from edges, features, and labels,
    then assign each a contiguous integer index (0 to N-1)."""

    all_nodes = set()

    # From edges
    all_nodes.update(edges_df["src"].unique())
    all_nodes.update(edges_df["dst"].unique())

    # From raw features
    if raw_df is not None:
        all_nodes.update(raw_df["node"].unique())

    # From labels
    if labels_df is not None:
        all_nodes.update(labels_df["node"].unique())

    # Sort for deterministic ordering
    sorted_nodes = sorted(all_nodes)
    mapping = {name: idx for idx, name in enumerate(sorted_nodes)}

    print(f"[OK] Built node mapping: {len(mapping)} unique nodes -> 0..{len(mapping)-1}")
    return mapping


# --------------------------------------------------------------------------
# Step 2: Build graph and compute structural features
# --------------------------------------------------------------------------
def build_graph_and_features(edges_df, node_mapping):
    """Build a NetworkX graph using numeric IDs and compute structural features."""

    G = nx.Graph()
    G.add_nodes_from(range(len(node_mapping)))  # ensure all nodes exist

    for _, row in edges_df.iterrows():
        u = node_mapping[row["src"]]
        v = node_mapping[row["dst"]]
        G.add_edge(u, v)

    print(f"[OK] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute structural features
    nodes = sorted(G.nodes())
    degree = dict(G.degree())
    pagerank = nx.pagerank(G)
    clustering = nx.clustering(G)

    structural_df = pd.DataFrame({
        "node": nodes,
        "degree": [degree[n] for n in nodes],
        "pagerank": [round(pagerank[n], 6) for n in nodes],
        "clustering_coeff": [round(clustering[n], 6) for n in nodes],
    })

    print(f"[OK] Computed structural features for {len(nodes)} nodes")
    return structural_df


# --------------------------------------------------------------------------
# Step 3: Load and map raw transactional data
# --------------------------------------------------------------------------
def load_and_map_raw_features(raw_path, node_mapping):
    """Load raw transactional features and map string node IDs to integers."""

    if not os.path.exists(raw_path):
        print(f"[!] No raw_features.csv found at {raw_path}")
        return None

    df = pd.read_csv(raw_path)

    # Map string IDs to integers
    df["node"] = df["node"].map(node_mapping)

    # Drop rows with unmapped nodes (shouldn't happen, but safety)
    before = len(df)
    df = df.dropna(subset=["node"])
    df["node"] = df["node"].astype(int)
    after = len(df)
    if before != after:
        print(f"[!] Dropped {before - after} rows with unmapped nodes from raw_features")

    print(f"[OK] Loaded raw transactional features: {len(df)} rows, columns = {list(df.columns)}")
    return df


# --------------------------------------------------------------------------
# Step 4: Convert edges to numeric
# --------------------------------------------------------------------------
def convert_edges(edges_df, node_mapping, output_path):
    """Map string edge IDs to numeric and write edges.csv."""

    numeric_edges = pd.DataFrame({
        "source": edges_df["src"].map(node_mapping),
        "target": edges_df["dst"].map(node_mapping),
    })
    numeric_edges = numeric_edges.astype(int)
    numeric_edges.to_csv(output_path, index=False)
    print(f"[OK] Wrote numeric edges.csv ({len(numeric_edges)} edges)")


# --------------------------------------------------------------------------
# Step 5: Convert labels to numeric
# --------------------------------------------------------------------------
def convert_labels(labels_df, node_mapping, output_path):
    """Map string label node IDs to numeric and write labels.csv."""

    if labels_df is None:
        print("[!] No labels.csv to convert")
        return

    numeric_labels = labels_df.copy()
    numeric_labels["node"] = numeric_labels["node"].map(node_mapping).astype(int)
    numeric_labels = numeric_labels.sort_values("node").reset_index(drop=True)
    numeric_labels.to_csv(output_path, index=False)
    print(f"[OK] Wrote numeric labels.csv ({len(numeric_labels)} rows)")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Fraud Detection - Feature Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load raw input files (with string IDs)
    print("\n--- Loading input files ---")
    edges_df = pd.read_csv(EDGES_CSV)
    print(f"[OK] Loaded edges.csv: {len(edges_df)} edges, columns = {list(edges_df.columns)}")

    raw_df = None
    if os.path.exists(RAW_FEATURES):
        raw_df = pd.read_csv(RAW_FEATURES)
        print(f"[OK] Loaded raw_features.csv: {len(raw_df)} rows")

    labels_df = None
    if os.path.exists(LABELS_CSV):
        labels_df = pd.read_csv(LABELS_CSV)
        print(f"[OK] Loaded labels.csv: {len(labels_df)} rows")

    # 2. Build string -> integer mapping
    print("\n--- Building node mapping ---")
    node_mapping = build_node_mapping(edges_df, raw_df, labels_df)

    # Save the mapping for reference
    mapping_df = pd.DataFrame(
        list(node_mapping.items()), columns=["original_id", "numeric_id"]
    )
    mapping_df.to_csv(OUTPUT_MAPPING, index=False)
    print(f"[OK] Saved node_mapping.csv ({len(mapping_df)} entries)")

    # 3. Build graph and compute structural features
    print("\n--- Computing structural features ---")
    structural_df = build_graph_and_features(edges_df, node_mapping)

    # 4. Map and merge transactional features
    print("\n--- Merging features ---")
    if raw_df is not None:
        mapped_raw = load_and_map_raw_features(RAW_FEATURES, node_mapping)
        merged = structural_df.merge(mapped_raw, on="node", how="left")
        merged = merged.fillna(0)
    else:
        merged = structural_df
        print("[!] No raw transactional data; using structural features only")

    # Sort by node ID for consistency
    merged = merged.sort_values("node").reset_index(drop=True)
    merged.to_csv(OUTPUT_FEATURES, index=False)
    print(f"[OK] Wrote features.csv ({len(merged)} rows, {len(merged.columns)} columns)")
    print(f"    Columns: {list(merged.columns)}")

    # 5. Convert edges to numeric
    print("\n--- Converting edges ---")
    convert_edges(edges_df, node_mapping, OUTPUT_EDGES)

    # 6. Convert labels to numeric
    print("\n--- Converting labels ---")
    convert_labels(labels_df, node_mapping, OUTPUT_LABELS)

    print("\n" + "=" * 60)
    print("  Done! All files written to ../data/")
    print("  Node IDs are now 0 to", len(node_mapping) - 1)
    print("=" * 60)


if __name__ == "__main__":
    main()
