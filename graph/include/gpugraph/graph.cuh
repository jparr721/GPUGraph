#pragma once

#include <vector>
#include <gpugraph/tensor.cuh>

namespace graph {
  // Graph Node
  typedef struct {
    unsigned int connections;
    tensor data;
  } g_node;

  struct graph_memory {
    float* graph;
    float* nodes;
    float* edges;
    int n_edges;
    int n_nodes;
  };

  class graph {
    public:
      graph & operator=(const graph&) = delete;
      graph(const graph&) = delete;
      explicit graph(size_t max_capacity);

      ~graph();

      void add_edge(int x1, int y1, int x2, int y2);
      void add_node(int x, int y, g_node data);
    private:
      // Our block size
      constexpr static int block_size = 64;

      // Initialize our device bound shared memory block
      graph_memory d_mem_;

      // Our host adjacency list
      std::vector<std::vector<int>> adjacencies_;
  };

  __global__ void kadd_edge(graph_memory mem, int x1, int y1, int x2, int y2);
  __global__ void kadd_node(graph_memory mem, int x, int y, float data);
} // namespace graph
