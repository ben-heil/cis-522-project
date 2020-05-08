# This script downloads data from the Broad bioimage benchmark collection and stores it locally

# We assume you're calling this script from the base of the repository. If not, update projectdir
# to contain a path to the repo base
PROJECTDIR=.
cd $PROJECTDIR
mkdir data 2> /dev/null

cd $PROJECTDIR/data
mkdir train
wget -nc https://storage.googleapis.com/rxrx/rxrx1.zip
unzip rxrx1.zip -d train

