#include <cassert>
#include <iostream>
#include <gpugraph/graph.cuh>

namespace graph {
  void graph::build() {
    assert(!is_built_);

    // We want to build after all nodes are added
    assert(!digraph_.empty());

    // Initialize our graph memory into the GPU
    cudaMalloc(&d_mem_.graph, d_mem_.n_nodes);

    // # of edges will be dynamic, but default to 3 * max_cap
    cudaMalloc(&d_mem_.edges, d_mem_.n_edges);
  }

  void graph::safe_add_node(const g_node& new_node) {
    auto found = digraph_.find(new_node);

    if (found == digraph_.end()) {
      digraph_.insert({ new_node, std::vector<g_node>() });
    }
  }

  void graph::unsafe_add_node(const g_node& new_node) {
    digraph_.insert({ new_node, std::vector<g_node>() });
  }

  void graph::add_edge(const g_node& source, const g_node& new_node) {
    // If graph is built, fail
    assert(!is_built_);

    digraph_[source].push_back(new_node);
  }

  graph::~graph() {
    // Clean up our graph memory
    cudaFree(d_mem_.graph);
    cudaFree(d_mem_.edges);
  }
} // namespace graph
