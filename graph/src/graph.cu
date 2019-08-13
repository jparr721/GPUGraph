#include <cassert>
#include <iostream>
#include <stack>
#include <utility>

#include <gpugraph/graph.cuh>

namespace graph {
  void graph::build() {
    assert(!is_built_);

    // We want to build after all nodes are added
    assert(!digraph_.empty());

    // Make our shared pointer
    digraph_ = std::make_shared<digraph>;
  }

  void graph::safe_add_node(const g_node& new_node) {
    bool found = false;
    for (int i = 0; i < digraph_->node_pos; ++i) {
      if (digraph_->nodes[i] != NULL && digraph_->nodes[i] == new_node) {
        found = true;
      }
    }

    if (!found && digraph_->node_pos < max_nodes) {
      digraph_->nodes[digraph_->node_pos] = new_node;
      ++digraph_->node_pos;
    }
  }

  void graph::unsafe_add_node(const g_node& new_node) {
    digraph_->nodes[digraph_->node_pos] = new_node;
  }

  void graph::add_edge(const g_node& source, const g_node& new_node) {
    // If graph is built, fail
    assert(!is_built_);

    // We don't want cycles between neural networks in this case,
    // always feed forward.
    if (!has_edge(source, new_node)) {
      int idx = find_key_pos(source);
      int new_idx = digraph_->nodes[idx].node_pos;

      digraph_->nodes[idx][++new_idx] = new_node;
      ++digraph_->nodes[idx].node_pos;
    }
  }

  // ========================================================
  bool graph::key_exists(const g_node& node) {
    for (int kk = 0; kk < digraph_->nodes[kk].node_pos; ++kk) {
      if (digraph_->nodes[kk].key == node) {
        return true;
      }
    }

    return false;
  }

  // ========================================================
  ///
  /// Searches for an edge in a depth-first manner
  ///
  bool graph::has_edge(const g_node& source, const g_node& destination) {
    std::stack<std::pair<g_node, in>> history;
    std::vector<g_node> visited;

    history.push({digraph_->nodes[idx], 0});

    int idx = 0;
    while !history.empty() {
      // TODO
    }
  }

  // ========================================================
  int graph::find_key_pos(const g_node& node) {
    for (int kk = 0; kk < digraph_->nodes[kk].node_pos; ++kk) {
      if (digraph_->nodes[kk] == node) {
        return kk;
      }
    }

    return -1;
  }

  // ========================================================

  graph::~graph() {}
} // namespace graph
