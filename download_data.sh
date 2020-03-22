# This script downloads data from the Broad bioimage benchmark collection and stores it locally

# We assume you're calling this script from the base of the repository. If not, update projectdir
# to contain a path to the repo base
PROJECTDIR=.
cd $PROJECTDIR
mkdir data 2> /dev/null

cd $PROJECTDIR/data
mkdir stage1_train
wget -nc https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
unzip stage1_train.zip -d stage1_train

mkdir stage1_test.zip
wget -nc https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip
unzip stage1_test.zip -d stage1_test

mkdir stage1_train_labels
wget -nc https://data.broadinstitute.org/bbbc/BBBC038/stage1_train_labels.csv

mkdir stage1_solution
wget -nc https://data.broadinstitute.org/bbbc/BBBC038/stage1_solution.csv
