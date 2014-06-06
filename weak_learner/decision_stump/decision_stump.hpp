/**
 * @file decision_stump.hpp
 * @author Udit Saxena
 * 
 * Defintion of decision stumps.
 */

#ifndef _MLPACK_METHODS_DECISION_STUMP_HPP
#define _MLPACK_METHODS_DECISION_STUMP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace decision_stump {

template <typename MatType = arma::mat>
class DecisionStump
{
 private:
  //other data types
  size_t numClass; // no of classes
  arma::Mat<size_t> spL;
  arma::Row<size_t> classLabels;
  arma::rowvec splittingCol;
  arma::mat split;
  size_t defaultClass;
  int splitCol; // splitting attribute
  int oneClass; // is 0 if multiple classes exist, else 1 if only one class
  size_t bucketSize;
  template <typename rType> int isDistinct(const arma::Row<rType>& featureRow);
  double SetupSplitAttribute(const arma::rowvec& attribute);
  double CalculateEntropy(const arma::rowvec& attribute, const arma::rowvec& labels);
  void TrainOnAtt(const arma::rowvec& attribute);
  template <typename rType> size_t CountMostFreq(const arma::Row<rType>& subCols);
  void MergeRanges();
 public:
  DecisionStump(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t classes, 
                size_t inpBucketSize);
  
  //classify function
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);
};

}; //namespace decision_stump
}; //namespace mlpack

#include "decision_stump_impl.cpp"

#endif