"""
Fast circle neighbor finder using NumPy + scipy KDTree.

Two circles are "neighbors" if they overlap or touch:
    distance(centers) <= r_i + r_j

CSV format expected:  x, y, radius  (with or without header)

Usage:
    python circle_neighbors.py circles.csv
    python circle_neighbors.py circles.csv --output neighbors.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def load_circles(path: str) -> np.ndarray:
    """Load CSV into (N, 3) array: [x, y, r]."""
    df = pd.read_csv(path, header=None)
    # Auto-detect header
    if df.iloc[0].dtype == object or not np.issubdtype(df.iloc[0].dtype, np.number):
        df = pd.read_csv(path)
    arr = df.values.astype(np.float64)
    assert arr.shape[1] >= 3, "CSV must have at least 3 columns: x, y, radius"
    return arr[:, :3]  # x, y, r


def find_neighbors(circles: np.ndarray) -> list[list[int]]:
    """
    Return adjacency list: neighbors[i] = sorted list of j where circles i and j overlap/touch.

    Strategy:
      1. Build KDTree on centers.
      2. For each circle i, query only circles within r_i + max_r  (upper-bound radius sum).
      3. Exact check: dist(i,j) <= r_i + r_j.

    Time complexity: O(N log N) average for sparse/uniform distributions.
    """
    xy = circles[:, :2]
    r  = circles[:, 2]
    max_r = r.max()

    tree = KDTree(xy)
    neighbors = [[] for _ in range(len(circles))]

    # Query radius per circle: r_i + max_r  (guaranteed superset of true neighbors)
    query_radii = r + max_r

    # Batch query
    candidate_lists = tree.query_ball_point(xy, query_radii, workers=-1)

    for i, candidates in enumerate(candidate_lists):
        if not candidates:
            continue
        cands = np.array(candidates, dtype=np.int64)
        # Remove self
        cands = cands[cands != i]
        if len(cands) == 0:
            continue

        # Vectorized exact check
        dx = xy[cands, 0] - xy[i, 0]
        dy = xy[cands, 1] - xy[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        radsum = r[cands] + r[i]

        true_neighbors = cands[dist <= radsum]
        neighbors[i] = true_neighbors.tolist()

    return neighbors


def main():
    parser = argparse.ArgumentParser(description="Find neighbors of each circle in a CSV.")
    parser.add_argument("csv", help="Input CSV file with columns: x, y, radius")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to CSV (columns: circle_id, neighbor_id)")
    args = parser.parse_args()

    print(f"Loading '{args.csv}' ...")
    circles = load_circles(args.csv)
    print(f"  {len(circles)} circles loaded.")

    print("Computing neighbors ...")
    neighbors = find_neighbors(circles)

    # Summary
    counts = [len(n) for n in neighbors]
    print(f"  Done. Neighbor count — min: {min(counts)}, "
          f"max: {max(counts)}, mean: {np.mean(counts):.2f}")

    if args.output:
        rows = [(i, j) for i, nb in enumerate(neighbors) for j in nb]
        out_df = pd.DataFrame(rows, columns=["circle_id", "neighbor_id"])
        out_df.to_csv(args.output, index=False)
        print(f"  Saved to '{args.output}'.")
    else:
        # Print first 10
        print("\nFirst 10 circles:")
        for i in range(min(10, len(neighbors))):
            print(f"  Circle {i}: {neighbors[i]}")

    return neighbors


if __name__ == "__main__":
    main()


# ── Importable API ────────────────────────────────────────────────────────────
# from circle_neighbors import load_circles, find_neighbors
#
# circles = load_circles("my_circles.csv")
# neighbors = find_neighbors(circles)   # list of lists
