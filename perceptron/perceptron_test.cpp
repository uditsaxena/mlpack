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

BOOST_AUTO_TEST_SUITE(PERCETRONTEST);

BOOST_AUTO_TEST_CASE(AND)
{
  mat trainData;
  trainData << 0 << 1 << 1 << 0 << endr
            << 1 << 0 << 1 << 0 <<endr;
  Mat<size_t> labels;
  labels << 1 << 1 << 1 << 0;

  Perceptron<> p(trainData, labels.row(0));
}
BOOST_AUTO_TEST_SUITE_END();