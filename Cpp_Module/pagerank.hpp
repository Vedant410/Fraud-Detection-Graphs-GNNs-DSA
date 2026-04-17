#ifndef PAGERANK_HPP
#define PAGERANK_HPP

#include "graph.hpp"
#include <unordered_map>

using namespace std;

unordered_map<int, double> computePageRank(Graph &g, int iterations = 20, double d = 0.85);

#endif