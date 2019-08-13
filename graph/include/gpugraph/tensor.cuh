#pragma once

namespace graph {
  constexpr int block_size = 64;

  typedef struct {
    unsigned int dim_x;
    unsigned int dim_y;
    unsigned int indices;
    float* buf;
  } tensor;

  void add_item(tensor &t, int x, int y, int z, float data);
  float get_item(const tensor& t, int x, int y, int z);

  __device__ void d_add_item(tensor t, int x, int y, float value);
  __device__ float d_get_item(tensor t, int x, int y);
  __device__ tensor d_get_sub_tensor(const tensor t, int row, int col);

  void tensor_hadmard_product(const tensor t1, const tensor t2, tensor output);
  __global__ void ktensor_hadmard_product(tensor t1, tensor t2, tensor output);

  size_t get_size(const tensor t);

} // namespace graph
