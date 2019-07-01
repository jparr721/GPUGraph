#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <optional>
#include <random>
#include <stdexcept>
#include <sstream>
#include <tuple>
#include <vector>

/// Tensors represent a H W D blob of data in memory
/// which will interact with the Tensorflow Tensor system
namespace graph { namespace types {
  template<typename Scalar_, unsigned Dimx, unsigned Dimy, unsigned Indices>
  class tensor {
    public:
      typedef Scalar_ Scalar;
      // Allocate our shared memory tensor
      std::array<Scalar, Dimx * Dimy * Indices> buf_;

      enum InitType {
        Zeroes = 0x00,
        Ones = 0x01,
        RandomUniform = 0x02,
        RandomNormal = 0x03,
      };

      explicit tensor(InitType it) { _initialize_tensor(it); }

      tensor<Scalar, Dimx, Dimy, Indices>
      operator+(tensor<Scalar, Dimx, Dimy, Indices>& other) {
        if (other.shape() == this->shape()) {
          throw std::runtime_error("Operands must be of the same shape");
        }
        for (auto i = 0u; i < this->tensor_size; ++i) {
          this->buf_[i] += other.buf_[i];
        }
      }

      tensor<Scalar, Dimx, Dimy, Indices>
      operator-(tensor<Scalar, Dimx, Dimy, Indices>& other) {
        if (other.shape() == this->shape()) {
          throw std::runtime_error("Operands must be of the same shape");
        }

        for (auto i = 0u; i < this->tensor_size; ++i) {
          this->buf_[i] -= other.buf_[i];
        }
      }

      /// Tensor Hadmard Product
      tensor<Scalar, Dimx, Dimy, Indices>
      operator*(tensor<Scalar, Dimx, Dimy, Indices>& other) {
        if (other.shape() == this->shape()) {
          throw std::runtime_error("Operands must be of the same shape");
        }
        for (auto i = 0u; i < this->tensor_size; ++i) {
          this->buf_[i] *= other.buf_[i];
        }
      }

      unsigned int rank() const { return Indices; }
      unsigned int size() const { return this->tensor_size; }
      std::tuple<unsigned, unsigned, unsigned>
      shape() const { return std::make_tuple(Dimx, Dimy, Indices); }

    private:
      unsigned int tensor_size = Dimx * Dimy * Indices;

      void _initialize_tensor(InitType it) {

        switch(it) {
          case 0x00: {
            auto init_val = 0;
            for (auto i = 0u; i < this->tensor_size; ++i) {
              buf_[i] = init_val;
            }
            break;
          }
          case 0x01: {
            auto init_val = 1;
            for (auto i = 0u; i < this->tensor_size; ++i) {
              buf_[i] = init_val;
            }
            break;
          }
          case 0x02: {
            std::default_random_engine gen;
            std::uniform_real_distribution<float> norm(0.0, 1.0);
            for (auto i = 0u; i < this->tensor_size; ++i) {
              auto init_val = norm(gen);
              buf_[i] = init_val;
            }
            break;
          }
          case 0x03: {
            std::default_random_engine gen;
            std::normal_distribution<float> norm(0.0, 1.0);
            for (auto i = 0u; i < this->tensor_size; ++i) {
              auto init_val = norm(gen);
              buf_[i] = init_val;
            }
            break;
          }
        }
      }
      void _zero_init() {}
      void _one_init() {}
      void _random_uniform_init() {}
      void _random_normal_init() {}

  };

} // namespace types
} // namespace graph
