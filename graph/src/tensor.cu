#include <cassert>
#include <gpugraph/tensor.cuh>

namespace graph {
  __device__ void tensor_add_item(tensor t, int x, int y, float value) {
    t.buf[x * t.indices + y] = value;
  }

  __device__ float tensor_get_item(tensor t, int x, int y) {
    return t.buf[x * t.indices + y];
  }

  __device__ tensor get_sub_tensor(const tensor t, int row, int col) {
    tensor t_sub;
    t_sub.dim_x = block_size;
    t_sub.dim_y = block_size;
    t_sub.indices = t.indices;
    t_sub.buf = &t.buf[t.indices * block_size * row + block_size * col];

  return t_sub;
  }

  void ktensor_hadmard_product(const tensor t1, const tensor t2, tensor output) {
    // Load our tensors into device memory
    tensor d_t1;
    d_t1.dim_x = t1.dim_x;
    d_t1.dim_y = t1.dim_y;
    d_t1.indices = t1.indices;

    size_t t1_tensor_size = get_size(t1);

    tensor d_t2;
    d_t2.dim_x = t2.dim_x;
    d_t2.dim_y = t2.dim_y;
    d_t2.indices = t2.indices;

    size_t t2_tensor_size = get_size(t2);

    // Make our output matrix
    tensor d_output;
    d_output.dim_x = output.dim_x;
    d_output.dim_y = output.dim_y;
    d_output.indices = output.indices;

    size_t output_tensor_size = get_size(output);


    assert(t1_tensor_size == t2_tensor_size);
    assert(output_tensor_size == t1_tensor_size
           && output_tensor_size == t2_tensor_size);

    // Malloc the space for our temp buffer
    cudaMalloc(&d_t1.buf, t1_tensor_size);
    cudaMalloc(&d_t2.buf, t2_tensor_size);
    cudaMalloc(&d_output, output_tensor_size);

    // Copy our host structures to the device structure
    cudaMemcpy(d_t1.buf, t1.buf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t2.buf, t2.buf, cudaMemcpyHostToDevice);

    // Invoke our hadmard product kernel
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(t2.dim_x / dim_block.x, t1.dim_y / dim_block.y);

    ktensor_hadmard_product<<<dim_grid, dim_block>>>(d_t1, d_t2, d_output);

    // Read from output device memory
    cudaMemcpy(
        output.buf, d_output.buf, output_tensor_size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_output);
  }

  __global__ void ktensor_hadmard_product(tensor t1, tensor t2, tensor output) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Each thread gets a subsection
    tensor t_sub = get_sub_tensor(output, block_row, block_col);

    // Our accumulator
    float output_value = 0;

    // Thread row and column within our output tensor
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all subtensors and compute their values
    #pragma unroll
    for (int ii = 0; ii < (t1.dim_x / block_size); ++ii) {
       // Get sub tensor of t1
      tensor t1_sub = get_sub_tensor(t1, block_row, ii);

      // Get sub tensor of t2
      tensor t2_sub = get_sub_tensor(t2, block_row, ii);

      // Store them in shared memory
      __shared__ float t1_s[block_size][block_size];
      __shared__ float t2_s[block_size][block_size];

      // Load the sub tensors from memory
      t1_s[row][col] = get_item(t1_sub, row, col);
      t2_s[row][col] = get_item(t2_sub, row, col);

      // Synchronize
      _syncThreads();

      // Do tensor multiplication
      for (int jj = 0; jj < block_size; ++jj) {
        output_value += t1_s[row][jj] * t2_s[jj][col];
      }

      // Synchronize
      _syncThreads();
    }
  }

  size_t get_size(const tensor t) {
    return t.dim_x * t.dim_y * t.indices;
  }
} // namespace graph
