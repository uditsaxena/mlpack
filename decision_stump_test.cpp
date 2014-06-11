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

BOOST_AUTO_TEST_CASE(OneClass)
{
  const char* trainSet = "ds_trainSet1.csv";
  const char* trainlabels = "ds_trainSetLabels1.csv";
  const char* testSet = "ds_testSet1.csv";
  size_t numClasses = 2;
  const char* output = "ds_output1.csv";
  size_t inpBucketSize = 6;

  mat trainingData;
  data::Load(trainSet, trainingData, true);
  
  Mat<size_t> labelsIn;
  data::Load(trainlabels, labelsIn, true);
  // no need to normalize labels here.

  mat testingData;
  data::Load(testSet, testingData, true);
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  for(int i =0; i < predictedLabels.size(); i++ )
    BOOST_CHECK_EQUAL(predictedLabels(i),1);  

} 

BOOST_AUTO_TEST_CASE(EntropyCalculation)
{
  const char* trainSet = "ds_trainSet3.csv";
  const char* trainlabels = "ds_trainSetLabels3.csv";
  const char* testSet = "ds_testSet3.csv";
  size_t numClasses = 2;
  const char* output = "ds_output3.csv";
  size_t inpBucketSize = 3;

  mat trainingData;
  data::Load(trainSet, trainingData, true);
  
  Mat<size_t> labelsIn;
  data::Load(trainlabels, labelsIn, true);
  // no need to normalize labels here.

  mat testingData;
  data::Load(testSet, testingData, true);
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);
  
  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);
  
  data::Save(output, predictedLabels, true, true);
}

BOOST_AUTO_TEST_CASE(PerfectSplitOnZero)
{
  const char* trainSet = "ds_trainSet4.csv";
  const char* trainlabels = "ds_trainSetLabels4.csv";
  const char* testSet = "ds_testSet4.csv";
  size_t numClasses = 2;
  const char* output = "ds_output4.csv";
  size_t inpBucketSize = 2;

  mat trainingData;
  data::Load(trainSet, trainingData, true);
  
  Mat<size_t> labelsIn;
  data::Load(trainlabels, labelsIn, true);
  // no need to normalize labels here.

  mat testingData;
  data::Load(testSet, testingData, true);
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}


BOOST_AUTO_TEST_CASE(BinningTesting)
{
  const char* trainSet = "ds_trainSet5.csv";
  const char* trainlabels = "ds_trainSetLabels5.csv";
  const char* testSet = "ds_testSet5.csv";
  size_t numClasses = 2;
  const char* output = "ds_output5.csv";
  size_t inpBucketSize = 10;

  mat trainingData;
  data::Load(trainSet, trainingData, true);
  
  Mat<size_t> labelsIn;
  data::Load(trainlabels, labelsIn, true);
  // no need to normalize labels here.

  mat testingData;
  data::Load(testSet, testingData, true);
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}


BOOST_AUTO_TEST_CASE(PerfectMultiClassSplit)
{
  const char* trainSet = "ds_trainSet6.csv";
  const char* trainlabels = "ds_trainSetLabels6.csv";
  const char* testSet = "ds_testSet6.csv";
  size_t numClasses = 4;
  const char* output = "ds_output6.csv";
  size_t inpBucketSize = 3;

  mat trainingData;
  data::Load(trainSet, trainingData, true);
  
  Mat<size_t> labelsIn;
  data::Load(trainlabels, labelsIn, true);
  // no need to normalize labels here.

  mat testingData;
  data::Load(testSet, testingData, true);
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}

BOOST_AUTO_TEST_CASE(MultiClassSplit)
{
  const char* trainSet = "ds_trainSet7.csv";
  const char* trainlabels = "ds_trainSetLabels7.csv";
  const char* testSet = "ds_testSet7.csv";
  size_t numClasses = 3;
  const char* output = "ds_output7.csv";
  size_t inpBucketSize = 3;

  mat trainingData;
  data::Load(trainSet, trainingData, true);
  
  Mat<size_t> labelsIn;
  data::Load(trainlabels, labelsIn, true);
  // no need to normalize labels here.

  mat testingData;
  data::Load(testSet, testingData, true);
  
  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  data::Save(output, predictedLabels, true, true);
}

BOOST_AUTO_TEST_SUITE_END();