#include <fstream>
#include "features.hpp"

void exportCSV(Graph &g, unordered_map<int, Features> &fmap) {
    ofstream edges("data/edges.csv");
    ofstream feats("data/features.csv");

    edges << "src,dst\n";
    feats << "node,degree,clustering,pagerank,txn_freq,total_amount, heuristic\n";

    for (auto &[u, edgesList] : g.adj) {
        for (auto &e : edgesList)
            edges << u << "," << e.to << "\n";
    }

    for (auto &[node, f] : fmap) {
        feats << node << "," << f.degree << "," << f.clustering << ","
              << f.pagerank << "," << f.txn_freq << "," << f.total_amount << "," << f.heuristic_score << "\n";
    }
}