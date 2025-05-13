# or wherever you want to save this
export DEEPCLEAN_CONTAINER_ROOT=~/images/deepclean
mkdir -p $DEEPCLEAN_CONTAINER_ROOT

export DATA_DIR=~/deepclean/data/CDC_test-120Hz
mkdir -p $DATA_DIR

export RESULTS_DIR=~/deepclean/results
mkdir -p $RESULTS_DIR

export DEEPCLEAN_IFO='H1'
export DEEPCLEAN_PROBLEM='120Hz'
export GPU_INDEX=0  # or whichever you want

#poetry run law run deepclean.tasks.Fetch     --data-dir $DATA_DIR     --start 1250899218     --end 1252108818     --sample-rate 4096      --min-duration 8192     --max-duration 32768     --image ~/images/deepclean/data.sif     --job-log fetch.log

#poetry run law run deepclean.tasks.Train   --image train.sif  --gpus $GPU_INDEX --data-fname $DATA_DIR/deepclean-1250899218-32768.hdf5 --output-dir $RESULTS_DIR/my-first-run
