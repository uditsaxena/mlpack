/*
 * @file: perceptron_test.cpp
 * @author: Udit Saxena
 */
#include <mlpack/core.hpp>
#include "perceptron.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Perceptron

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::perceptron;
using namespace std;

BOOST_AUTO_TEST_SUITE(PERCETRONTEST);

BOOST_AUTO_TEST_CASE(AND)
{
  cout<<"\nAnd Case: \n";
  mat trainData;
  trainData << 0 << 1 << 1 << 0 << endr
            << 1 << 0 << 1 << 0 << endr;
  Mat<size_t> labels;
  labels << 0 << 0 << 1 << 0;

  Perceptron<> p(trainData, labels.row(0));

  mat testData;
  testData << 0 << 1 << 1 << 0 << endr
           << 1 << 0 << 1 << 0 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);
}

BOOST_AUTO_TEST_CASE(OR)
{
  cout<<"\nOR Case: \n";
  mat trainData;
  trainData << 0 << 1 << 1 << 0 << endr
            << 1 << 0 << 1 << 0 << endr;

  Mat<size_t> labels;
  labels << 1 << 1 << 1 << 0;

  Perceptron<> p(trainData, labels.row(0));

  mat testData;
  testData << 0 << 1 << 1 << 0 << endr
            << 1 << 0 << 1 << 0 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);
}

BOOST_AUTO_TEST_CASE(RANDOM3)
{
  cout<<"\nOR Case: \n";
  mat trainData;
  trainData << 0 << 1 << 1 << 4 << 5 << 4 << 1 << 2 << 1 << endr
           << 1 << 0 << 1 << 1 << 1 << 2 << 4 << 5 << 4 << endr;

  Mat<size_t> labels;
  labels << 0 << 0 << 0 << 1 << 1 << 1 << 2 << 2 << 2;

  Perceptron<> p(trainData, labels.row(0));

  mat testData;
  testData << 0 << 1 << 1 << endr
           << 1 << 0 << 1 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);
}
BOOST_AUTO_TEST_SUITE_END();