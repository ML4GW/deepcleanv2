# or wherever you want to save this
export DEEPCLEAN_CONTAINER_ROOT=~/images/deepclean
mkdir -p $DEEPCLEAN_CONTAINER_ROOT

export DATA_DIR=~/deepclean/data/CDC_test
mkdir -p $DATA_DIR

export RESULTS_DIR=~/deepclean/results
mkdir -p $RESULTS_DIR

export DEEPCLEAN_IFO='H1'
export DEEPCLEAN_PROBLEM='30Hz'
export GPU_INDEX=0  # or whichever you want
