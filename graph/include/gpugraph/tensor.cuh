#pragma once

#include <eigen3/Eigen/Dense>

namespace graph { namespace types {

  template<typename Scalar_, unsigned Indices_>
  class tensor {
    public:
      enum InitType {
        Zeroes = this->zero_init;
        Ones = this->one_init;
        RandomUniform = this->random_uniform_init;
        RandomNormal = this->random_normal_init;
      };

      unsigned Indices_ indices;
      typedef Scalar_ Scalar;
      explicit tensor(InitType it) {

      }

      unsigned int rank() const { return indices; }

      void zero_init();
      void one_init();
      void random_uniform_init();
      void random_normal_init();

    private:

  }

} // namespace types
} // namespace graph
