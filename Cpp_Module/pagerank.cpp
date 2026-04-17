#include "graph.hpp"
#include <unordered_map>

using namespace std;

unordered_map<int, double> computePageRank(Graph &g, int iterations = 20, double d = 0.85) {
    unordered_map<int, double> pr;
    int N = g.adj.size();

    for (auto &[node, _] : g.adj) pr[node] = 1.0 / N;

    for (int it = 0; it < iterations; it++) {
        unordered_map<int, double> new_pr;

        for (auto &[node, _] : g.adj)
            new_pr[node] = (1 - d) / N;

        for (auto &[u, edges] : g.adj) {
            if (edges.empty()) continue;

            double share = pr[u] / edges.size();
            for (auto &e : edges)
                new_pr[e.to] += d * share;
        }

        pr = new_pr;
    }

    return pr;
}