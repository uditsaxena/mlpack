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
  arma::mat tempWeights(uniqueLabels.n_elem, data.n_rows);// = arma::randu<MatType>
  tempWeights.fill(0.0);
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

  trainData.print("This is the trainData matrix: ");
  classLabels.print("This is the classLabels matrix: ");
  weightVectors.print("This is the weightVectors matrix: ");

  int iterations = 2, j, i = 0, flag = 0;
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
    flag = 1;
    // Now this inner loop is for going through the dataset in each iteration
    for (j = 0; j < data.n_cols; j++)
    {
      // flag = 1;
      tempLabelMat = weightVectors * data.col(j);

      // tempLabelMat.print("In the iterations, value of tempLabelMat: ");

      maxVal = tempLabelMat.max(maxIndexRow, maxIndexCol);
      // labels.print("Value of labels: ");
      if(maxIndexRow != labels(0,j))
      {
        flag = 0;
        tempLabel = labels(0,j);

        std::cout<<"Not Equal. The value of maxIndexRow is "<<maxIndexRow
                 <<"\nAnd the value of tempLabel is: "<<tempLabel
                 <<"\nValue of maxVal is : "<<maxVal<<"\n";
        // send max index row for knowing which weight to update, 
        // send j to know the value of the vector to update it with.
        // send templabel to know the correct class 
        UpdateWeights(maxIndexRow, j, tempLabel);
      }
    }
  }
  weightVectors.print("final value of weightVectors is : ");
}

/*The classification function.*/
template <typename MatType>
void Perceptron<MatType>::Classify(const MatType& test, 
                                   arma::Row<size_t>& predictedLabels)
{
  int i;
  arma::mat tempLabelMat;
  arma::uword maxIndexRow, maxIndexCol;
  double maxVal;
  for (i = 0; i < test.n_cols; i++)
  {
    tempLabelMat = weightVectors * test.col(i);
    maxVal = tempLabelMat.max(maxIndexRow, maxIndexCol);

    predictedLabels(0,i) = maxIndexRow;
  }
  predictedLabels.print("Value of predictedLabels: ");
}

template <typename MatType>
void Perceptron<MatType>::UpdateWeights(size_t rowIndex, size_t labelIndex, size_t vectorIndex)
{
  MatType instance = trainData.col(labelIndex);
  instance.print("\nvalue of f; ");
  weightVectors.print("Value of weightVectors before update: ");
  weightVectors.row(rowIndex) = weightVectors.row(rowIndex) - 
                               instance.t();

  weightVectors.row(vectorIndex) = weightVectors.row(vectorIndex) + 
                                 instance.t();
  weightVectors.print("Value of weightVectors after update: ");

  // updating like so: 
  // for correct class : w = w + x
  // for incorrect class : w = w - x
};

} // namespace perceptron
} // namespace mlpack

#endif