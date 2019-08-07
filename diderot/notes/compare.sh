#!/bin/bash
# Usage compare target-directory 
if [ "$#" -ne 1 ]
then
  echo "Usage: compare.sh <target directory>"
  echo "Example: compare.sh ./tmp/"
  echo "Target directory should contain all chapters and their xmls"
  echo "This utility will then pairwise compare same xmls"
  exit 1
fi

echo "diffing data_collection"
diff data_collection.xml $1/data_collection.xml


echo "yyy"
diff yyy.xml $1/yyy.xml

echo "debugging"
diff debugging.xml $1/debugging.xml

echo "free_text"
diff free_text.xml $1/free_text.xml

echo "graphs"
diff graphs.xml $1/graphs.xml

echo "intro"
diff intro.xml $1/intro.xml

echo "linear_classification"
diff linear_classification.xml $1/linear_classification.xml

echo "matrices"
diff matrices.xml $1/matrices.xml

echo "ml_intro"
diff ml_intro.xml $1/ml_intro.xml

echo "mle"
diff mle.xml $1/mle.xml

echo "nonlinear_modeling"
diff nonlinear_modeling.xml $1/nonlinear_modeling.xml

echo "probability"
diff probability.xml $1/probability.xml

echo "recommender"
diff recommender.xml $1/recommender.xml

echo "relational_data"
diff relational_data.xml $1/relational_data.xml

echo "unsupervised"
diff unsupervised.xml $1/unsupervised.xml

echo "visualization"
diff visualization.xml $1/visualization.xml
