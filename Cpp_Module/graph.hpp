#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <unordered_map>
#include <vector>

using namespace std;

struct Edge {
    int to;
    double amount;
    int timestamp;
};

class Graph {
public:
    unordered_map<int, vector<Edge>> adj;

    void addEdge(int u, int v, double amount, int time);
};

#endif