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
  /*
  This function is called to update the weightVectors matrix. It uses the 
  Gradient Descent method to correct the 
  
  @param: trainData - the training dataset.
  @param: weightVectors - matrix of weight vectors.
  @param: classLabels - labels of input vectors.
  @param: rowIndex - index of the row which has been incorrectly predicted.
  @param: labelIndex - index of the vector in trainData.
  @param: vectorIndex - index of the class which should have been predicted.
 */
  void UpdateWeights(const arma::mat& trainData, arma::mat& weightVectors,
                size_t labelIndex, size_t vectorIndex, size_t rowIndex = 0)
  { 
    double eta = 0.05;
    arma::mat instance = trainData.col(labelIndex); // this is x
  
    // weightVectors.row(rowIndex) = weightVectors.row(rowIndex) - 
    //                            instance.t();

    weightVectors.row(vectorIndex) = weightVectors.row(vectorIndex) + 
                                 eta * (vectorIndex - 
                                 weightVectors.row(vectorIndex) * instance) * instance.t() ;
  }
};
}; // namespace perceptron
}; // namespace mlpack

#endif