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
    defaultClass = classLabels(0); 
    // put some indicator of some kind here.
  }
  else
  {
    oneClass = 0;
    int bestAtt=-1,i,j;
    double entropy,bestEntropy=DBL_MAX; 
    // setting default values of splitting attribute
    
    defaultClass = CountMostFreq<size_t>(classLabels);

    // std::cout<<"Okay. Entered oneClass as "<<oneClass<<"\n";

    for (i = 0;i < data.n_rows;i++)
    {
      if (isDistinct<double>(data.row(i)))
      {
        // std::cout<<"Entering SetupSplitAttribute with i as :"<<i<<" and the data row as:\n";
        // data.row(i).print();

        entropy=SetupSplitAttribute(data.row(i));

        // std::cout<<"Value of entropy "<<entropy<<"\n";
    
        if( entropy < bestEntropy )
        {
          bestAtt = i;
          bestEntropy = entropy;
        } 

      }
    }

    // std::cout<<"Entropy calculation done with value of bestEntropy : "
             // <<bestEntropy<<" and the bestAtt as: "<<bestAtt<<"\n";
  
    splitCol = bestAtt;

    TrainOnAtt(data.row(splitCol));

  }

}

template <typename MatType>
double DecisionStump<MatType>::SetupSplitAttribute(const arma::rowvec& attribute)
{
  // std::cout<<"Okay. Now entering SetupSplitAttribute \n";

  int i, count, begin, end;
  double entropy = 0.0;

  arma::rowvec sortedAtt = arma::sort(attribute);
  // sortedAtt.print("Value of the sorted Att: ");

  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());
  // sortedIndexAtt.print("Value of sorted Index Att: ");

  // ^ index of sorted elements.
  arma::Row<size_t> sortedLabels(attribute.n_elem,arma::fill::zeros);
  
  // classLabels.print("Unsorted class labels: ");
  // std::cout<<"Now going to sort into classes \n";

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedIndexAtt(i));
  
  // sortedLabels.print("Value of sorted Labels is: ");

  // sortedLabels now has the labels, sorted as per the attribute.

  arma::rowvec subCols;
  arma::rowvec subColAtts;

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

      // subCols.print("These are the subCols on which entropy is being calculated.");

      subColAtts = sortedAtt.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedAtt.cols(begin, end)).n_elem);

      entropy += CalculateEntropy(subColAtts, subCols);
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
  // std::cout<<"Okay. Now entering TrainOnAttribute \n";

  int i, count, begin, end;

  arma::rowvec sortedSplitAtt = arma::sort(attribute);
  arma::uvec sortedSplitIndexAtt = arma::stable_sort_index(attribute.t());
  arma::Row<size_t> sortedLabels(attribute.n_elem,arma::fill::zeros);
  arma::mat tempSplit;

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedSplitIndexAtt(i));
  
  // std::cout<<"Now going to learn.\n";
  // sortedLabels.print("Value of sorted Labels is: ");
  
  arma::rowvec subCols;
  int mostFreq;
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
      // subCols.print("subCols are: ");
      
      mostFreq = CountMostFreq<double>(subCols);
      
      // std::cout<<"Most Frequent value is: "<<mostFreq<<"\n";

      tempSplit << begin << mostFreq << arma::endr;
      split = arma::join_cols(split, tempSplit);
      i = end + 1;
      count = 0;
    }
    else
      i++;
  }
  split.print("value of final split");
  // another function to quietly process and merge ranges.
  MergeRanges();
  // 
}

template <typename MatType>
void DecisionStump<MatType>::MergeRanges()
{
  // split.print("This is the initial value of split, on entering MergeRanges(). ");
  int i;
  for (i = 1;i < split.n_rows; i++)
  {
    if (split(i,1) == split(i-1,1))
    {
      // remove this row, 
      // carry on to the next one ?
      // how, exactly ?
      split.shed_row(i);
      i--; // go back to previous row.
    }
  }
  // split.print("This is the final value of split, when leaving MergeRanges(). ");

}

template <typename MatType>
template <typename rType>
size_t DecisionStump<MatType>::CountMostFreq(const arma::Row<rType>& subCols)
{
  arma::Row<rType> sortCounts = arma::sort(subCols);
  // sortCounts.print("Value of sortCounts: ");

  int count = 0, localCount = 0,i;

  for (i = 0; i < sortCounts.n_elem ; ++i)
  {
    if (i == sortCounts.n_elem - 1)
    {
      if (sortCounts(i-1) == sortCounts(i))
        localCount++;
      if (localCount > count)
        count = localCount;
    }
    else if (sortCounts(i) != sortCounts(i+1))
    {
      localCount = 0;
      count++;
    }
    else
    {
      localCount++;
      if (localCount > count)
        count = localCount;
    }
  }
  return count;
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
double DecisionStump<MatType>::CalculateEntropy(const arma::rowvec& attribute,
                                                const arma::rowvec& labels)
{
  int i,j,count;
  double entropy=0.0;
  
  // std::cout<<"Entering CalculateEntropy.\n";

  arma::rowvec uniqueAtt = arma::unique(attribute);
  arma::rowvec uniqueLabel = arma::unique(labels);
  arma::Row<size_t> numElem(uniqueAtt.n_elem,arma::fill::zeros); 
  arma::Mat<size_t> entropyArray(uniqueAtt.n_elem,numClass,arma::fill::zeros); 
  
  // uniqueAtt.print("Value of uniqueAtt is: ");
  // uniqueLabel.print("Value of uniqueLabel is: ");

  for (j = 0;j < uniqueAtt.n_elem; j++)
  {
    for (i = 0; i < attribute.n_elem; i++)
    {
      if (uniqueAtt[j] == attribute[i])
      {
        entropyArray(j,labels(i))++;
        numElem(j)++;
      }
    }
  }

  // entropyArray.print("Again, value of entropyArray is: ");

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
  
  // std::cout<<"Value of Entropy inside CalculateEntropy() is: "<<entropy<<"\n";

  return entropy;
}

template<typename MatType>
void DecisionStump<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels)
{
  int i,j,val;
  if ( !oneClass )
  {
    for (i = 0; i < test.n_cols; ++i)
    {
      val = test(splitCol,i);
      for ( j = 0; j < split.n_rows; j++)
      {
        if (j == split.n_rows - 1)
          predictedLabels(i) = split(j,1);

        else if ( (val >= split(j,0)) && (val < split(j + 1,0)) )
          predictedLabels(i) = split(j,1);

        // else put default class (majority class)
        else
          predictedLabels(i) = defaultClass; 

      }
    }
  }
  else
  {
    for (i = 0;i < test.n_cols;i++)
      predictedLabels(i)=defaultClass;
  }

}
}; // namespace decision_stump
}; // namespace mlpack

#endif