#include "graph.hpp"
#include "features.hpp"
#include "heuristics.hpp"

extern unordered_map<int, double> computePageRank(Graph &g);

int main() {
    Graph g;

    // Example (replace with file input)
    g.addEdge(1, 2, 100, 1);
    g.addEdge(1, 3, 200, 2);

    auto fmap = computeFeatures(g);

    auto pr = computePageRank(g);
    for (auto &[node, val] : pr)
        fmap[node].pagerank = val;
    computeHeuristicScores(fmap);
    exportCSV(g, fmap);

    return 0;
}