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
private:
  arma::Row<size_t> classLabels;
  arma::mat weightVectors,trainData;
  void UpdateWeights(size_t rowIndex, size_t labelIndex);
};
} // namespace perceptron
} // namespace mlpack

#include "perceptron_impl.cpp"

#endif