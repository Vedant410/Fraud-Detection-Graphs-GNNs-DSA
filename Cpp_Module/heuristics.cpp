#include "heuristics.hpp"
#include <algorithm>

// Normalize helper
double normalize(double val, double min_val, double max_val) {
    if (max_val - min_val == 0) return 0;
    return (val - min_val) / (max_val - min_val);
}

void computeHeuristicScores(unordered_map<int, Features> &fmap) {
    double max_deg = 0, max_pr = 0, max_cluster = 0, max_freq = 0;

    // Find max values for normalization
    for (auto &[node, f] : fmap) {
        max_deg = max(max_deg, f.degree);
        max_pr = max(max_pr, f.pagerank);
        max_cluster = max(max_cluster, f.clustering);
        max_freq = max(max_freq, f.txn_freq);
    }

    // Compute heuristic score
    for (auto &[node, f] : fmap) {
        double deg_norm = normalize(f.degree, 0, max_deg);
        double pr_norm = normalize(f.pagerank, 0, max_pr);
        double cluster_norm = normalize(f.clustering, 0, max_cluster);
        double freq_norm = normalize(f.txn_freq, 0, max_freq);

        // Weighted scoring (tune these weights later)
        double score =
            0.30 * deg_norm +
            0.25 * pr_norm +
            0.25 * cluster_norm +
            0.20 * freq_norm;

        f.heuristic_score = score;
    }
}