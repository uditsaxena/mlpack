/*
 * @file: perceptron.hpp
 * @author: Udit Saxena
 *
 *
 * Definition of Perceptron
 */

#ifndef _MLPACK_METHODS_PERCEPTRON_HPP
#define _MLPACK_METHODS_PERCEPTRON_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {

template <typename MatType = arma::mat>
class Perceptron
{
public:
  //Constructor
  Perceptron(const MatType& data, const arma::Row<size_t>& labels);
  // The classification function.
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);
private:
  arma::Row<size_t> classLabels;
  arma::mat weightVectors,trainData, biasVector;
  void UpdateWeights(size_t rowIndex, size_t labelIndex, size_t vectorIndex);
};
} // namespace perceptron
} // namespace mlpack

#include "perceptron_impl.cpp"

#endif