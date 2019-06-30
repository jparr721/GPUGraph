#pragma once

// Fix for gcc compat with CUDA
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>
#include <utility>

namespace graph {
  class graph {
    public:
      graph & operator=(const graph&) = delete;
      graph(const graph&) = delete;
      graph() = default;

      ~graph();

      void add_edge(std::pair<int, int> edge);
      void add_node(Eigen::MatrixXf node);

    private:
      std::vector<Eigen::MatrixXf> nodes;
  };

  class Node {
    public:
      Node(const Eigen::MatrixXf weights, const std::string& signature);

  };
} // namespace graph
