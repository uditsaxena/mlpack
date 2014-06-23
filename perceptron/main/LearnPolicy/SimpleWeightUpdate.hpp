/*
 *  @file: SimpleWeightUpdate.hpp
 *  @author: Udit Saxena
 *
 */

#ifndef _MLPACK_METHOD_PERCEPTRON_LEARN_SIMPLEWEIGHTUPDATE
#define _MLPACK_METHOD_PERCEPTRON_LEARN_SIMPLEWEIGHTUPDATE

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {

class SimpleWeightUpdate 
{
public:
  SimpleWeightUpdate()
  { }

  void UpdateWeights(const arma::mat& trainData, arma::mat& weightVectors,
                const arma::Row<size_t>& classLabels, size_t rowIndex, 
                size_t labelIndex, size_t vectorIndex)
  {
    arma::mat instance = trainData.col(labelIndex);
  
    weightVectors.row(rowIndex) = weightVectors.row(rowIndex) - 
                               instance.t();

    weightVectors.row(vectorIndex) = weightVectors.row(vectorIndex) + 
                                 instance.t();
  }
};
}; // namespace perceptron
}; // namespace mlpack

#endif