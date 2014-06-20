/*
 *  @file: zeroinit.cpp
 *  @author: Udit Saxena
 *
 */

#ifndef _MLPACK_METHOS_PERCEPTRON_ZEROINIT
#define _MLPACK_METHOS_PERCEPTRON_ZEROINIT

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {
  class ZeroInitialization
  {
  public:
    ZeroInitialization()
    { }

    inline static void initialize(arma::mat& W, size_t row, size_t col)
    {
      arma::mat tempWeights(row, col);
      tempWeights.fill(0.0);

      W = tempWeights;
    }
  }; // class ZeroInitialization
}; // namespace perceptron
}; // namespace mlpack

#endif