#include "features.hpp"
#include <unordered_set>

// Degree
double computeDegree(Graph &g, int node) {
    return g.adj[node].size();
}

// Transaction Frequency (edges per node)
double computeFrequency(Graph &g, int node) {
    return g.adj[node].size();
}

// Total transaction amount
double computeAmount(Graph &g, int node) {
    double sum = 0;
    for (auto &e : g.adj[node]) sum += e.amount;
    return sum;
}

// Clustering Coefficient
double clusteringCoeff(Graph &g, int node) {
    auto &neighbors = g.adj[node];
    int k = neighbors.size();
    if (k < 2) return 0;

    unordered_set<int> neighborSet;
    for (auto &e : neighbors) neighborSet.insert(e.to);

    int links = 0;
    for (auto &e : neighbors) {
        for (auto &x : g.adj[e.to]) {
            if (neighborSet.count(x.to)) links++;
        }
    }
    return (double)links / (k * (k - 1));
}

unordered_map<int, Features> computeFeatures(Graph &g) {
    unordered_map<int, Features> fmap;

    for (auto &[node, _] : g.adj) {
        Features f;
        f.degree = computeDegree(g, node);
        f.txn_freq = computeFrequency(g, node);
        f.total_amount = computeAmount(g, node);
        f.clustering = clusteringCoeff(g, node);
        f.pagerank = 0; // filled later

        fmap[node] = f;
    }
    return fmap;
}