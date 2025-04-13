# or wherever you want to save this
export DEEPCLEAN_CONTAINER_ROOT=~/deepClean/dc-review/images/deepclean
mkdir -p $DEEPCLEAN_CONTAINER_ROOT

export DATA_DIR=~/deepClean/dc-review/deepclean/data
mkdir -p $DATA_DIR

export RESULTS_DIR=~/deepClean/dc-review/deepclean/results
mkdir -p $RESULTS_DIR

export DEEPCLEAN_IFO='H1'
export GPU_INDEX=0  # or whichever you want

#poetry run law run deepclean.tasks.Fetch     --data-dir $DATA_DIR     --start 1250899218     --end 1252108818     --sample-rate 4096      --min-duration 8192     --max-duration 32768     --image ~/images/deepclean/data.sif     --job-log fetch.log

#poetry run law run deepclean.tasks.Train   --image train.sif  --gpus $GPU_INDEX --data-fname $DATA_DIR/deepclean-1250899218-32768.hdf5 --output-dir $RESULTS_DIR/my-first-run
