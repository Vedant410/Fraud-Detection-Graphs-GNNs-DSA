#include "graph.hpp"

void Graph::addEdge(int u, int v, double amount, int time) {
    adj[u].push_back({v, amount, time});
    adj[v]; // ensure node exists
}