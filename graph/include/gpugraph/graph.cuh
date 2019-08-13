#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>

#include <gpugraph/tensor.cuh>

namespace graph {
  constexpr int max_nodes = 20;
  constexpr int max_connections = 40;

  // Graph Nodes represent an entire neural network weight structure
  struct g_node {
    std::string name;
    tensor data;
  };

  struct node {
    g_node key;
    g_node connections[max_connections];
    int node_pos;
  };

  struct digraph {
    node nodes[max_nodes];
  };

  class graph {
    public:
      graph & operator=(const graph&) = delete;
      graph(const graph&) = delete;
      graph() {}

      ~graph();

      void add_edge(const g_node& source, const g_node& new_node);

      void safe_add_node(const g_node& new_node);
      void unsafe_add_node(const g_node& new_node);

      void build();

      bool built() { return is_built_; }

      digraph get_digraph() { return digraph_; }
    private:
      // Our block size
      constexpr static int block_size = 64;

      bool is_built_ = false;

      // Make our adjacency list
      std::unique_ptr<digraph> digraph_;

      // ========================================================
      int find_key_pos(const g_node& new_node);

      // ========================================================
      bool key_exists(const g_node& new_node);

      // ========================================================
      bool has_edge(const g_node& source, const g_node& destination);
  };
} // namespace graph
