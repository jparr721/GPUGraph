#include <iostream>
#include <gpugraph/graph.cuh>

namespace graph {
  graph::graph(size_t max_capacity) {
    // Initialize our graph memory into the GPU
    cudaMalloc(&d_mem_.graph, max_capacity);

    // # Nodes will be the graph size
    cudaMalloc(&d_mem_.nodes, max_capacity);

    // # Edges will be dynamic, but default to 3 * max_cap
    cudaMalloc(&d_mem_.edges, 3 * max_capacity);
  }

  graph::~graph() {
    // Clean up our graph memory
    cudaFree(d_mem_.graph);
    cudaFree(d_mem_.nodes);
    cudaFree(d_mem_.edges);
  }
} // namespace graph
