#include <iostream>
#include <gpugraph/graph.cuh>

namespace graph {
  void graph::add_edge(std::pair<int, int> edge) {
    int x = std::get<0>(edge);
    int y = std::get<1>(edge);

    edges_(x, y) = 1;
  }

  void graph::add_node(Eigen::MatrixXf node) {
    if (nodes_.size() < max_capacity_) {
      nodes_.push_back(node);
    } else {
      // TODO add a logging lib
      std::cout << "Graph at capacity" << std::endl;
    }
  }
} // namespace graph
