/**
 * @file decision_stump_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of DecisionStump class.
 */

#ifndef __MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_IMPL_HPP
#define __MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_IMPL_HPP

// In case it hasn't been included yet.
#include "decision_stump.hpp"

#include <set>
#include <algorithm>

namespace mlpack {
namespace decision_stump {

/**
 * Constructor. Train on the provided data. Generate a decision stump from data.
 *
 * @param data Input, training data.
 * @param labels Labels of data.
 * @param classes Number of distinct classes in labels.
 * @param inpBucketSize Minimum size of bucket when splitting.
 */
template<typename MatType>
DecisionStump<MatType>::DecisionStump(const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const size_t classes,
                                      size_t inpBucketSize)
{
  numClass = classes;
  bucketSize = inpBucketSize;

  // If classLabels are not all identical, proceed with training.
  int bestAtt = -1;
  double entropy;
  double bestEntropy = DBL_MAX;

  // Set the default class to handle attribute values which are not present in
  // the training data.
  //defaultClass = CountMostFreq<size_t>(classLabels);

  for (int i = 0; i < data.n_rows; i++)
  {
    // Go through each attribute of the data.
    if (isDistinct<double>(data.row(i)))
    {
      // For each attribute with non-identical values, treat it as a potential
      // splitting attribute and calculate entropy if split on it.
      entropy = SetupSplitAttribute(data.row(i), labels);

      // Find the attribute with the bestEntropy so that the gain is
      // maximized.
      if (entropy < bestEntropy)
      {
        bestAtt = i;
        bestEntropy = entropy;
      }

      /* This section is commented out because I believe entropy calculation is
       * wrong.  Entropy should only be 0 if there is only one class, in which
       * case classification is perfect and we can take the shortcut below.

      // If the entropy is 0, then all the labels are the same and we are done.
      Log::Debug << "Entropy is " << entropy << "\n";
      if (entropy == 0)
      {
        // Only one split element... there is no split at all, just one bin.
        split.set_size(1);
        binLabels.set_size(1);
        split[0] = -DBL_MAX;
        binLabels[0] = labels[0];
        splitCol = 0; // It doesn't matter.
        return;
      }
      */
    }
  }
  splitCol = bestAtt;

  // Once the splitting column/attribute has been decided, train on it.
  TrainOnAtt<double>(data.row(splitCol), labels);
}

/**
 * Classification function. After training, classify test, and put the predicted
 * classes in predictedLabels.
 *
 * @param test Testing data or data to classify.
 * @param predictedLabels Vector to store the predicted classes after
 *      classifying test
 */
template<typename MatType>
void DecisionStump<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels)
{
  for (int i = 0; i < test.n_cols; i++)
  {
    // Determine which bin the test point falls into.
    // Assume first that it falls into the first bin, then proceed through the
    // bins until it is known which bin it falls into.
    int bin = 0;
    const double val = test(splitCol, i);

    while (bin < split.n_elem - 1)
    {
      if (val < split(bin + 1))
        break;

      ++bin;
    }

    predictedLabels(i) = binLabels(bin);
  }
}

/**
 * Sets up attribute as if it were splitting on it and finds entropy when
 * splitting on attribute.
 *
 * @param attribute A row from the training data, which might be a candidate for
 *      the splitting attribute.
 */
template <typename MatType>
double DecisionStump<MatType>::SetupSplitAttribute(
    const arma::rowvec& attribute,
    const arma::Row<size_t>& labels)
{
  int i, count, begin, end;
  double entropy = 0.0;

  // Sort the attribute in order to calculate splitting ranges.
  arma::rowvec sortedAtt = arma::sort(attribute);

  // Store the indices of the sorted attribute to build a vector of sorted
  // labels.  This sort is stable.
  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());

  arma::Row<size_t> sortedLabels(attribute.n_elem);
  sortedLabels.fill(0);

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = labels(sortedIndexAtt(i));

  i = 0;
  count = 0;
  double ratioEl;
  // This splits the sorted into buckets of size greater than or equal to
  // inpBucketSize.
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      // if we're at the end, then don't worry about the bucket size
      // just take this as the last bin.
      begin = i - count + 1;
      end = i;
      ratioEl = ((double)(end - begin + 1)/sortedLabels.n_elem);
      // std::cout<<"\nRatio of Elements: "<<ratioEl<<"\n";
      entropy += ratioEl * CalculateEntropy<double, size_t>(
                 sortedAtt.subvec(begin,end),sortedLabels.subvec(begin,end));
      i++;
    }
    else if (sortedLabels(i) != sortedLabels(i + 1))
    {
      // if we're not at the last element of sortedLabels, then check whether
      // count is less than the current bucket size.
      if (count < bucketSize)
      {
        // if it is, then take the minimum bucket size anyways
        begin = i - count + 1;
        end = begin + bucketSize - 1;

        if (end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        // if it is not, then take the bucket size as the value of count.
        begin = i - count + 1;
        end = i;
      }
      ratioEl = ((double)(end - begin + 1)/sortedLabels.n_elem);
      // std::cout<<"\nRatio of Elements: "<<ratioEl<<"\n";
      entropy +=ratioEl * CalculateEntropy<double, size_t>(
                 sortedAtt.subvec(begin,end),sortedLabels.subvec(begin,end));

      i = end + 1;
      count = 0;
    }
    else
      i++;
  }
  // std::cout<<"Final value of entropy: "<<entropy<<"\n";
  return entropy;
}

/**
 * After having decided the attribute on which to split, train on that
 * attribute.
 *
 * @param attribute Attribute is the attribute decided by the constructor on
 *      which we now train the decision stump.
 */
template <typename MatType>
template <typename rType>
void DecisionStump<MatType>::TrainOnAtt(const arma::rowvec& attribute,
                                        const arma::Row<size_t>& labels)
{
  int i, count, begin, end;

  arma::rowvec sortedSplitAtt = arma::sort(attribute);
  arma::uvec sortedSplitIndexAtt = arma::stable_sort_index(attribute.t());
  arma::Row<size_t> sortedLabels(attribute.n_elem);
  sortedLabels.fill(0);
  arma::vec tempSplit;
  arma::Row<size_t> tempLabel;

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = labels(sortedSplitIndexAtt(i));

  arma::rowvec subCols;
  rType mostFreq;
  i = 0;
  count = 0;
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      begin = i - count + 1;
      end = i;

      arma::rowvec zSubCols((sortedLabels.cols(begin, end)).n_elem);
      zSubCols.fill(0.0);

      subCols = sortedLabels.cols(begin, end) + zSubCols;

      mostFreq = CountMostFreq<double>(subCols);

      split.resize(split.n_elem + 1);
      split(split.n_elem - 1) = sortedSplitAtt(begin);
      binLabels.resize(binLabels.n_elem + 1);
      binLabels(binLabels.n_elem - 1) = mostFreq;

      i++;
    }
    else if (sortedLabels(i) != sortedLabels(i + 1))
    {
      if (count < bucketSize)
      {
        // Test for different values of bucketSize, especially extreme cases.
        begin = i - count + 1;
        end = begin + bucketSize - 1;

        if (end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        begin = i - count + 1;
        end = i;
      }
      arma::rowvec zSubCols((sortedLabels.cols(begin, end)).n_elem);
      zSubCols.fill(0.0);

      subCols = sortedLabels.cols(begin, end) + zSubCols;

      // Find the most frequent element in subCols so as to assign a label to
      // the bucket of subCols.
      mostFreq = CountMostFreq<double>(subCols);//sortedLabels.subvec(begin, end));

      split.resize(split.n_elem + 1);
      split(split.n_elem - 1) = sortedSplitAtt(begin);
      binLabels.resize(binLabels.n_elem + 1);
      binLabels(binLabels.n_elem - 1) = mostFreq;

      i = end + 1;
      count = 0;
    }
    else
      i++;
  }

  // Now trim the split matrix so that buckets one after the after which point
  // to the same classLabel are merged as one big bucket.
  MergeRanges();
}

/**
 * After the "split" matrix has been set up, merge ranges with identical class
 * labels.
 */
template <typename MatType>
void DecisionStump<MatType>::MergeRanges()
{
  for (int i = 1; i < split.n_rows; i++)
  {
    if (binLabels(i) == binLabels(i - 1))
    {
      // Remove this row, as it has the same label as the previous bucket.
      binLabels.shed_row(i);
      split.shed_row(i);
      // Go back to previous row.
      i--;
    }
  }
}

template <typename MatType>
template <typename rType>
rType DecisionStump<MatType>::CountMostFreq(const arma::Row<rType>& subCols)
{
  // Sort subCols for easier processing.
  arma::Row<rType> sortCounts = arma::sort(subCols);
  rType element;
  int count = 0, localCount = 0;

  // An O(n) loop which counts the most frequent element in sortCounts
  for (int i = 0; i < sortCounts.n_elem; ++i)
  {
    if (i == sortCounts.n_elem - 1)
    {
      if (sortCounts(i - 1) == sortCounts(i))
      {
        // element = sortCounts(i - 1);
        localCount++;
      }
      else if (localCount > count)
        count = localCount;
    }
    else if (sortCounts(i) != sortCounts(i + 1))
    {
      localCount = 0;
      count++;
    }
    else
    {
      localCount++;
      if (localCount > count)
      {
        count = localCount;
        if (localCount == 1)
          element = sortCounts(i);
      }
    }
  }
  return element;
}

/**
 * Returns 1 if all the values of featureRow are not same.
 *
 * @param featureRow The attribute which is checked for identical values.
 */
template <typename MatType>
template <typename rType>
int DecisionStump<MatType>::isDistinct(const arma::Row<rType>& featureRow)
{
  if (featureRow.max() - featureRow.min() > 0)
    return 1;
  else
    return 0;
}

/**
 * Calculate entropy of attribute.
 *
 * @param attribute The attribute for which we calculate the entropy.
 * @param labels Corresponding labels of the attribute.
 */
template<typename MatType>
template<typename AttType, typename LabelType>
double DecisionStump<MatType>::CalculateEntropy(arma::subview_row<AttType> attribute,
                                                arma::subview_row<LabelType> labels)
{
  double entropy = 0.0;
  size_t j;
  // labels.print("Value of the labels");
  // arma::rowvec uniqueAtt = arma::unique(attribute);
  // arma::Row<LabelType> uniqueLabel = arma::unique(labels);
  arma::Row<size_t> numElem(numClass); //uniqueAtt.n_elem);
  numElem.fill(0);
  // arma::Mat<size_t> entropyArray(uniqueAtt.n_elem,numClass);
  // entropyArray.fill(0);

  // Populate entropyArray and numElem; they are used as helpers to calculate
  // entropy.
  for (j = 0; j < labels.n_elem; j++)
  {
    numElem(labels(j))++;
    /*for (int i = 0; i < attribute.n_elem; i++)
    {
      if (uniqueAtt[j] == attribute[i])
      {
        entropyArray(j, labels(i))++;
        numElem(j)++;
      }
    }*/
  }
  // const double p1 = ((double)labels.n_elem / )
  // do this when the function call goes back.
  for (j = 0; j < numClass; j++)
  {
    // const double p1 = ((double) numElem(j) / attribute.n_elem);
    const double p1 = ((double) numElem(j) / labels.n_elem);
    // std::cout<<"Value of p1: "<<p1<<"\n";
    entropy += (p1 == 0) ? 0 : p1 * log2(p1);
    // std::cout<<"Value of entropy is : "<<entropy<<std::endl;
    /*for (int i = 0; i < numClass; i++)
    {
      const double p2 = ((double) entropyArray(j, i) / numElem(j));
      const double p3 = (p2 == 0) ? 0 : p2 * log2(p2);

      entropy += p1 * p3;
    }*/
  }
  // std::cout<<"Value of entropy for this bucket is : "<<entropy<<std::endl;
  return entropy;
}

}; // namespace decision_stump
}; // namespace mlpack

#endif
