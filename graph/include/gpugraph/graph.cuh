#pragma once

#include <vector>

namespace graph {
  struct graph_memory {
    const float** graph;
    const float** nodes;
    const float** edges;
    int edges;
    int nodes;
  };

  class graph {
    public:
      graph & operator=(const graph&) = delete;
      graph(const graph&) = delete;
      explicit graph(size_t max_capacity);

      ~graph();

      void add_edge(int x1, int y1, int x2, int y2);
      void add_node(int x, int y, float data);
      __global__ void kadd_edge(graph_memory mem, int x1, int y1, int x2, int y2);
      __global__ void kadd_node(graph_memory mem, int x, int y, float data);
    private:
      // Initialize our device bound shared memory block
      graph_memory mem_;

      // Our host adjacency list
      std::vector<std::vector<int>> adjacencies_;
  }
} // namespace graph
