/*
 * @author: Udit Saxena
 * @file: decision_stump_main.cpp
 *
 *
 */

#include <mlpack/core.hpp>
#include "decision_stump.hpp"

using namespace mlpack;
using namespace mlpack::decision_stump;
using namespace std;
using namespace arma;

PROGRAM_INFO("","");

// necessary parameters
PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING_REQ("labels_file", "A file containing labels for the training set.",
  "l");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");
PARAM_STRING_REQ("num_classes","The number of classes","c");

// output parameters (optional)
PARAM_STRING("output", "The file in which the predicted labels for the test set"
    " will be written.", "o", "output.csv");

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  const string trainingDataFilename = CLI::GetParam<string>("train_file");
  mat trainingData;
  data::Load(trainingDataFilename, trainingData, true);

  const string labelsFilename = CLI::GetParam<string>("labels_file");
  // Load labels.
  mat labelsIn;
  data::Load(labelsFilename, labelsIn, true);

  // helpers for normalizing the labels
  Col<size_t> labels;
  vec mappings;

  // Do the labels need to be transposed?
  if (labelsIn.n_rows == 1)
    labelsIn = labelsIn.t();

  // test the dimensionality of labels and training data as well
  /*Code goes here*/
  
  // normalize the labels
  data::NormalizeLabels(labelsIn.unsafe_col(0), labels, mappings);

  const size_t num_classes = CLI::GetParam<size_t>("num_classes");
  /*
  Should number of classes be input or should it be 
  derived from the labels row ?
  */
  const string testingDataFilename = CLI::GetParam<std::string>("test_file");
  mat testingData;
  data::Load(testingDataFilename, testingData, true);

  if (testingData.n_rows != trainingData.n_rows)
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << trainingData.n_rows - 1
        << ")!" << std::endl;

  // should labels be transposed here ? Are they being sent as column here ?

  Timer::Start("training");
  DecisionStump<> ds(trainingData,labels,num_classes);
  Timer::Stop("training");

  Row<size_t> predictedLabels(testingData.n_cols);
  Timer::Start("testing");
  ds.Classify(testingData, predictedLabels);
  Timer::Stop("testing");

  vec results;
  data::RevertLabels(predictedLabels, mappings, results);

  const string outputFilename = CLI::GetParam<string>("output");
  data::Save(outputFilename, results, true, true);
  // saving the predictedLabels in the transposed manner in output

  return 0;
}