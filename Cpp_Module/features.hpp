#ifndef FEATURES_HPP
#define FEATURES_HPP

#include "graph.hpp"
#include <unordered_map>

struct Features {
    double degree;
    double clustering;
    double pagerank;
    double txn_freq;
    double total_amount;
    double heuristic_score;
};

unordered_map<int, Features> computeFeatures(Graph &g);

#endif