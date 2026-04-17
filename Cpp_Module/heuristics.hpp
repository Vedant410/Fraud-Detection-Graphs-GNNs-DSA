#ifndef HEURISTICS_HPP
#define HEURISTICS_HPP

#include "features.hpp"
#include <unordered_map>

using namespace std;

void computeHeuristicScores(unordered_map<int, Features> &fmap);

#endif