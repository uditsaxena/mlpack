/*
 *  @file: GradientDescent.hpp
 *  @author: Udit Saxena
 *
 */

#ifndef _MLPACK_METHOD_PERCEPTRON_LEARN_GRADIENTDESCENT
#define _MLPACK_METHOD_PERCEPTRON_LEARN_GRADIENTDESCENT

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {
 
class GradientDescent 
{ 
public:
  GradientDescent()
  { }

  void UpdateWeights(const arma::mat& trainData, arma::mat& weightVectors,
                const arma::Row<size_t>& classLabels, size_t rowIndex, 
                size_t labelIndex, size_t vectorIndex)
  { 
    double eta = 0.1;
    arma::mat instance = trainData.col(labelIndex); // this is x
  
    // weightVectors.row(rowIndex) = weightVectors.row(rowIndex) - 
    //                            instance.t();

    weightVectors.row(vectorIndex) = weightVectors.row(vectorIndex) + 
                                 eta * (classLabels(0,labelIndex) - 
                                 weightVectors.row(vectorIndex) * instance) * instance.t() ;
  }
};
}; // namespace perceptron
}; // namespace mlpack

#endif