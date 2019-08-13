#pragma once

#include <string>
#include <map>
#include <vector>

#include <gpugraph/tensor.cuh>

namespace graph {
  // Graph Node
  typedef struct {
    std::string name;
    unsigned int connections;
    tensor data;
  } g_node;

  struct graph_memory {
    float* graph;
    float* edges;
    int n_edges;
    int n_nodes;
  };

  class graph {
    public:
      using adjacency_list = std::map<g_node, std::vector<g_node>>;

      graph & operator=(const graph&) = delete;
      graph(const graph&) = delete;
      explicit graph(size_t max_capacity) : max_capacity_(max_capacity) {}

      ~graph();

      void add_edge(const g_node& source, const g_node& new_node);

      void safe_add_node(const g_node& new_node);
      void unsafe_add_node(const g_node& new_node);

      void build();

      bool built() { return is_built_; }

      adjacency_list get_digraph() { return digraph_; }
    private:
      // Our block size
      constexpr static int block_size = 64;

      bool is_built_ = false;

      // Initialize our device bound shared memory block
      graph_memory d_mem_;

      // Make our adjacency list
      adjacency_list digraph_;

      size_t max_capacity_ = 0;
  };

  __global__ void kadd_edge(graph_memory mem, int x1, int y1, int x2, int y2);
  __global__ void kadd_node(graph_memory mem, int x, int y, float data);
} // namespace graph
