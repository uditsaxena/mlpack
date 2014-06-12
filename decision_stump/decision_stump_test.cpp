/*
 *  @file decision_stump_test.cpp
 *  @author Udit Saxena
 *  
 *  Test for Decision Stump  
 */

#include <mlpack/core.hpp>
#include "weak_learner/decision_stump/decision_stump.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DecisionStump
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace decision_stump;
using namespace arma;

BOOST_AUTO_TEST_SUITE(DSTEST);

/*
This tests handles the case wherein only one class exists in the input labels.
It checks whether the only class supplied was the only class predicted.
 */
BOOST_AUTO_TEST_CASE(OneClass)
{
  size_t numClasses = 2;
  size_t inpBucketSize = 6;

  mat trainingData;
  trainingData << 2.4 << 3.8 << 3.8 << endr
               << 1 << 1 << 2 << endr
               << 1.3 << 1.9 << 1.3 << endr;
  
  Mat<size_t> labelsIn;
  labelsIn << 1 << 1 << 1;
  
  // no need to normalize labels here.

  mat testingData;
  testingData << 2.4 << 2.5 << 2.6;
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  for(int i = 0; i < predictedLabels.size(); i++ )
    BOOST_CHECK_EQUAL(predictedLabels(i),1);  

} 

/*
This tests for the classification: 
 if testinput < 0 - class 0
 if testinput > 0 - class 1
An almost perfect split on zero.
*/
BOOST_AUTO_TEST_CASE(PerfectSplitOnZero)
{
  size_t numClasses = 2;
  const char* output = "outputPerfectSplitOnZero.csv";
  size_t inpBucketSize = 2;

  mat trainingData;
  trainingData << -1 << 1 << -2 << 2 << -3 << 3;
  
  Mat<size_t> labelsIn;
  labelsIn << 0 << 1 << 0 << 1 << 0 << 1;
  // no need to normalize labels here.

  mat testingData;
  testingData << -4 << 7 << -7 << -5 << 6;
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}

/*
This tests the binning function for the case when a dataset with 
cardinality of input < inpBucketSize is provided.
*/
BOOST_AUTO_TEST_CASE(BinningTesting)
{
  size_t numClasses = 2;
  const char* output = "outputBinningTesting.csv";
  size_t inpBucketSize = 10;

  mat trainingData;
  trainingData << -1 << 1 << -2 << 2 << -3 << 3 << -4;
 
  Mat<size_t> labelsIn;
  labelsIn << 0 << 1 << 0 << 1 << 0 << 1 << 0;
  
  // no need to normalize labels here.

  mat testingData;
  testingData << 5;
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}

/*
This is a test for the case when non-overlapping, multiple
classes are provided. It tests for a perfect split due to the
non-overlapping nature of the input classes.
*/
BOOST_AUTO_TEST_CASE(PerfectMultiClassSplit)
{
  size_t numClasses = 4;
  const char* output = "outputPerfectMultiClassSplit.csv";
  size_t inpBucketSize = 3;

  mat trainingData;
  trainingData << -8 << -7 << -6 << -5 << -4 << -3 << -2 << -1
               << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7;
  
  Mat<size_t> labelsIn;
  labelsIn << 0 << 0 << 0 << 0 << 1 << 1 << 1 << 1 
           << 2 << 2 << 2 << 2 << 3 << 3 << 3 << 3;
  // no need to normalize labels here.

  mat testingData;
  testingData << -6.1 << -2.1 << 1.1 << 5.1;
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}

/*
This test is for the case when reasonably overlapping, multiple classes 
are provided in the input label set. It tests whether classification 
takes place with a reasonable amount of error due to the overlapping 
nature of input classes.
*/
BOOST_AUTO_TEST_CASE(MultiClassSplit)
{
  size_t numClasses = 3;
  const char* output = "outputMultiClassSplit.csv";
  size_t inpBucketSize = 3;

  mat trainingData;
  trainingData << -7 << -6 << -5 << -4 << -3 << -2 << -1 << 0 << 1 
               << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9 << 10;
  
  Mat<size_t> labelsIn;
  labelsIn << 0 << 0 << 0 << 0 << 1 << 1 << 0 << 0 
           << 1 << 1 << 1 << 2 << 1 << 2 << 2 << 2 << 2 << 2;
  // no need to normalize labels here.

  mat testingData;
  testingData << -6.1 << -5.9 << -2.1 << -0.7 << 2.5 << 4.7 << 7.2 << 9.1;
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}

BOOST_AUTO_TEST_SUITE_END();