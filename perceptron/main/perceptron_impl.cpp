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

/*
  Constructor - Constructs the perceptron. Or rather, builds the weightVectors
  matrix, which is later used in Classification. 
  It adds a bias input vector of 1 to the input data to take care of the bias
  weights.

  @param: data - Input, training data.
  @param: labels - Labels of dataset.
   @param: iterations - maximum number of iterations the perceptron
                       learn algorithm is to be run.
*/
template <typename LearnPolicy, typename WeightInitializationPolicy, typename MatType>
Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Perceptron(const MatType& data,
                                const arma::Row<size_t>& labels, int iterations)
{
  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  WeightInitializationPolicy WIP;
  WIP.initialize(weightVectors, uniqueLabels.n_elem, data.n_rows + 1);
  
  // Start training.
  classLabels = labels; 

  trainData = data;
  // inserting a row of 1's at the top of the training data set.
  MatType zOnes(1, data.n_cols);
  zOnes.fill(1);
  trainData.insert_rows(0, zOnes);

  int j, i = 0, converged = 0;
  size_t tempLabel; 
  arma::uword maxIndexRow, maxIndexCol;
  double maxVal;
  arma::mat tempLabelMat;

  LearnPolicy LP;

  while ((i < iterations) && (!converged))
  {
    // This outer loop is for each iteration, 
    // and we use the 'converged' variable for noting whether or not
    // convergence has been reached.
    i++;
    converged = 1;

    // Now this inner loop is for going through the dataset in each iteration
    for (j = 0; j < data.n_cols; j++)
    {
      // Multiplying for each variable and checking 
      // whether the current weight vector correctly classifies this.
      tempLabelMat = weightVectors * trainData.col(j);

      maxVal = tempLabelMat.max(maxIndexRow, maxIndexCol);

      //checking whether prediction is correct.
      if(maxIndexRow != classLabels(0,j))
      {
        // due to incorrect prediction, convergence set to 0
        converged = 0;
        tempLabel = labels(0,j);

        // send maxIndexRow for knowing which weight to update, 
        // send j to know the value of the vector to update it with.
        // send tempLabel to know the correct class 
        LP.UpdateWeights(trainData, weightVectors, 
                         j, tempLabel, maxIndexRow);
      }
    }
  }
}

/*
  Classification function. After training, use the weightVectors matrix to 
  classify test, and put the predicted classes in predictedLabels.

  @param: test - testing data or data to classify. 
  @param: predictedLabels - vector to store the predicted classes after
                            classifying test
 */
template <typename LearnPolicy, typename WeightInitializationPolicy, typename MatType>
void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Classify(
                const MatType& test, arma::Row<size_t>& predictedLabels)
{
  int i;
  arma::mat tempLabelMat;
  arma::uword maxIndexRow, maxIndexCol;
  double maxVal;
  MatType testData = test;
  
  MatType zOnes(1, test.n_cols);
  zOnes.fill(1);
  testData.insert_rows(0, zOnes);
  
  for (i = 0; i < test.n_cols; i++)
  {
    tempLabelMat = weightVectors * testData.col(i);
    maxVal = tempLabelMat.max(maxIndexRow, maxIndexCol);

    predictedLabels(0,i) = maxIndexRow;
  }
}

}; // namespace perceptron
}; // namespace mlpack

#endif