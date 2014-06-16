/*
 *  @file: perceptron_impl.hpp
 *  @author: Udit Saxena
 *
 */

#ifndef _MLPACK_METHODS_PERCEPTRON_IMPL_CPP
#define _MLPACK_METHODS_PERCEPTRON_IMPL_CPP

#include "perceptron.hpp"

namespace mlpack {
namespace perceptron {

template <typename MatType>
Perceptron<MatType>::Perceptron(const MatType& data,
                                const arma::Row<size_t>& labels)
{
  arma::Row<size_t> uniqueLabels = arma::unique(labels);
  // a case for random initialization.
  arma::mat tempWeights = arma::randu<MatType>(uniqueLabels.n_elem, data.n_rows);
  // now that the matrix has been initialized, start training.
  
  arma::Row<size_t> zLabels(labels.n_elem);
  zLabels.fill(0);
  classLabels = labels + zLabels;

  MatType zData(data);
  zData.fill(0);
  trainData = data + zData;

  arma::mat zWeights(tempWeights);
  zWeights.fill(0.0);
  weightVectors = tempWeights + zWeights;

  // store labels and tempweight matrix into class variables.
  // these can be avoided if UpdateWeights is not a separate function.

  int iterations = 100, j, i = 0, flag = 0;
  size_t tempLabel; 
  arma::uword maxIndexRow, maxIndexCol;
  double maxVal;
  arma::mat tempLabelMat;

  while ((i < iterations) && (!flag))
  {
    // This outer loop is for each iteration, 
    // and we use the flag variable for noting whether or not
    // convergence has been reached.
    i++;
    // Now this inner loop is for going through the dataset in each iteration
    for (j = 0; j < data.n_cols; j++)
    {
      flag = 1;
      tempLabelMat = tempWeights * data.col(j);
      maxVal = tempLabelMat.max(maxIndexRow, maxIndexCol);

      if(maxIndexRow != labels(0,j))
      {
        flag = 0;
        tempLabel = labels(0,j);
        //send into another function which updates as required.
        // std::cout<<"Not equal !";
        // send max index row for knowing which weight to update, 
        // send tempLabel to know the value of the vector to update it with.
        UpdateWeights(maxIndexRow, tempLabel);
      }
    }
  }
  weightVectors.print("final value of weightVectors is : ");
}
template <typename MatType>
void Perceptron<MatType>::UpdateWeights(size_t rowIndex, size_t labelIndex)
{
  MatType instance = trainData.col(labelIndex);
  weightVectors.row(rowIndex) = weightVectors.row(rowIndex) - 
                               instance.t();

  weightVectors.row(labelIndex) = weightVectors.row(labelIndex) - 
                                 instance.t();

  // updating like so: 
  // for correct class : w = w + x
  // for incorrect class : w = w - x
};

} // namespace perceptron
} // namespace mlpack

#endif