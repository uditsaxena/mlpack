/**
 * @file decision_stump_impl.hpp
 * @author Udit Saxena
**/

#ifndef _MLPACK_METHODS_DECISION_STUMP_IMPL_HPP
#define _MLPACK_METHODS_DECISION_STUMP_IMPL_HPP

#include "decision_stump.hpp"

#include <set>
#include <algorithm>

namespace mlpack {
namespace decision_stump {

template<typename MatType>
DecisionStump<MatType>::DecisionStump(const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const size_t classes,
                                      size_t inpBucketSize) // talk about a def val for inpBucketSize
{
  classLabels = labels + arma::zeros<arma::Row<size_t> >(labels.n_elem);
  // opitmizing now
  numClass = classes;
  bucketSize = inpBucketSize;

  // Constructor
  if ( !isDistinct<size_t>(labels) )
  {
    oneClass = 1; 
    // put some indicator of some kind here.
  }
  else
  {
    oneClass = 0;
    int bestAtt=-1;
    double entropy,bestEntropy=DBL_MAX; 
    // setting default values of splitting attribute
    
    std::cout<<"Okay. Entered oneClass as "<<oneClass<<"\n";

    for (int i = 0;i < data.n_rows;i++)
    {
      if (isDistinct<double>(data.row(i)))
      {
        std::cout<<"Entering SetupSplitAttribute with i as :"<<i<<" and the data row as:\n";
        data.row(i).print();

        entropy=SetupSplitAttribute(data.row(i));

        std::cout<<"Value of entropy "<<entropy<<"\n";
    
        if( entropy < bestEntropy )
        {
          bestAtt = i;
          bestEntropy = entropy;
        } 

      }
    }

    std::cout<<"Entropy calculation done !\n";

    if ( bestAtt != -1 )
    {
      int i,j;  
      splitCol = bestAtt;

      arma::rowvec uniqueAtt = arma::unique(data.row(splitCol));
      arma::Mat<size_t> splitLabels(uniqueAtt.n_elem,numClass,arma::fill::zeros); 
      
      for (j = 0; j < uniqueAtt.size(); j++)
      {
        for (i = 0; i < data.n_cols; i++)
        {
          if (uniqueAtt(j) == data(splitCol,i))
          {
            splitLabels(j,labels(i))++;
          }
        }
      }
      spL = splitLabels + arma::zeros<arma::Mat<size_t> >(uniqueAtt.n_elem,numClass);
      splittingCol = data.row(splitCol) + arma::zeros<arma::rowvec >(data.n_cols);
      // get this to some local/class variable, which can be used by 
      // classify later on.
    }
      
    else
    {
      // no attributes to split on
    }
  }

}

template <typename MatType>
double DecisionStump<MatType>::SetupSplitAttribute(const arma::rowvec& attribute)
{
  std::cout<<"Okay. Now entering SetupSplitAttribute \n";

  int i, count, begin, end;
  double entropy = 0.0;

  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());
  sortedIndexAtt.print("Value of sorted Index Att: ");

  // ^ index of sorted elements.
  arma::Row<size_t> sortedLabels(attribute.n_elem,arma::fill::zeros);
  
  std::cout<<"Now going to sort into classes \n";

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedIndexAtt(i));
  
  sortedLabels.print("Value of sorted Labels is: ");

  // sortedLabels now has the labels, sorted as per the attribute.
  arma::rowvec subCols;
  // now start creating buckets:
  i = 0;
  count = 0;
  while (i < sortedLabels.n_elem - 1)
  {
    count++;
    if( sortedLabels(i) != sortedLabels(i + 1) )
    {
      if (count < bucketSize) // test for differevalues of bucketSize, especially extreme cases. 
      {
        begin = i - count + 1;
        end = begin + bucketSize - 1;
        
        if ( end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        begin = i - count + 1;
        end = i;
      }
      subCols = sortedLabels.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedLabels.cols(begin, end)).n_elem);

      subCols.print("These are the subCols on which entropy is being calculated.");

      entropy += CalculateEntropy(subCols);
      //sortedLabels.cols(begin, end));
      // has to be a better way around this - if not this then it throws
      // a problem with the sub_matrix view.

      // also, for training, you need to start storing 
      // either one of begin or end
      i = end + 1;
      count = 0;
    }
    else
      i++;
  }
  // buckets have been generated and sent to the relevant place.

  return entropy;
}

template <typename MatType>
void DecisionStump<MatType>::TrainOnAtt(const arma::rowvec& attribute)
{
  arma::rowvec splitPoints;
  int i, count, begin, end;

  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());

  arma::Row<size_t> sortedLabels(attribute.n_elem,arma::fill::zeros);

  arma::rowvec subCols;

  i = 0;
  count = 0;
  while (i < sortedLabels.n_elem - 1)
  {
    count++;
    if( sortedLabels(i) != sortedLabels(i + 1) )
    {
      if (count < bucketSize) // test for differevalues of bucketSize, especially extreme cases. 
      {
        begin = i - count + 1;
        end = begin + bucketSize - 1;
        
        if ( end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        begin = i - count + 1;
        end = i;
      }
      i = end + 1;
      count = 0;
    }
    else
      i++;
  }


}

template <typename MatType>
template <typename rType>
int DecisionStump<MatType>::isDistinct(const arma::Row<rType>& featureRow)
{
  if (featureRow.max()-featureRow.min() > 0)
    return 1;
  else
    return 0;
}

template<typename MatType>
double DecisionStump<MatType>::CalculateEntropy(const arma::rowvec& attribute)
{
  int i,j,count;
  double entropy=0.0;
  
  std::cout<<"Entering CalculateEntropy.\n";

  arma::rowvec uniqueAtt = arma::unique(attribute);
  arma::Row<size_t> numElem(uniqueAtt.n_elem,arma::fill::zeros); 
  arma::Mat<size_t> entropyArray(uniqueAtt.n_elem,numClass,arma::fill::zeros); 
  
  uniqueAtt.print("Value of uniqueAtt is: ");

  for (j = 0;j < uniqueAtt.n_elem; j++)
  {
    for (i = 0; i < attribute.n_elem; i++)
    {
      if (uniqueAtt[j] == attribute[i])
      {
        entropyArray(j,attribute(i))++;
        numElem(j)++;
      }
    }
  }

  entropyArray.print("Again, value of entropyArray is: ");

  double p1, p2, p3;
  for ( j = 0; j < uniqueAtt.size(); j++ )
  {
    p1 = ((double)numElem(j) / attribute.n_elem);

    for ( i = 0; i < numClass; i++)
    {
      p2 = ((double)entropyArray(j,i) / numElem(j));
      
      if(p2 == 0)
        p3 = 0;
      else
        p3 = (  p2 * log2(p2) );

      entropy+=( p1 * p3 );
    }
  }
  
  std::cout<<"Value of Entropy inside CalculateEntropy() is: "<<entropy<<"\n";

  return entropy;
}

template<typename MatType>
void DecisionStump<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels)
{
  int i,j;
  if ( !oneClass )
  {
    double max;
    arma::rowvec uniqueAtt = arma::unique(splittingCol);//trainData.row(splitCol));
      
    // now predict using majority voting
    arma::Row<size_t> splitClass(uniqueAtt.size());
    arma::Row<size_t> count; // helper row, to help find the max value;

    // for ( j = 0; j < uniqueAtt.size(); j++)
    // {
    //   // has to be a better way to get around this...
    //   count = spL.row(j);
    //   max = count.max(splitClass(j));
    //   // finding index of the class and then predicting that class.
    // }

    for ( i = 0; i < test.n_cols; i++)
    {
      for ( j = 0; j < uniqueAtt.size(); j++ )
      {
        if ( test(splitCol,i)==uniqueAtt[j] )
          predictedLabels(i)=splitClass[j];
      }
    }
  }

  else
  {
    for (i = 0;i < test.n_cols;i++)
      predictedLabels(i)=classLabels(0);
  }

}
}; // namespace decision_stump
}; // namespace mlpack

#endif