# Instructions to run deepcleanv2:main

## Setup
1. `mkdir dc-review`
2. `cd dc-review`
3. `git clone git@github.com:ML4GW/deepcleanv2.git`
4. `cd deepcleanv2`
5. `git submodule update --init --recursive`
6. `poetry install`
7. `cd -`
8. `mkdir -p images/deepclean`
9. `mkdir -p deepclean/data deepclean/results`

## Fetching Data
1. `cd deepcleanv2`
2. Setting deepcleanv2/export_vars.sh
    - DEEPCLEAN_CONTAINER_ROOT=\${HOME}/dc-review/images/deepclean
    - DATA_DIR=\${HOME}/dc-review/deepclean/data
    - RESULTS_DIR=${HOME}/dc-review/deepclean/results
3. `. export_vars.sh`
4. Setting deepcleanv2/luigi.cfg
    - Setting subtraction problem, the problems can be found in /deepclean/couplings.
5. Setting deepcleanv2/fetch.sh
6. `. fetch.sh`

## Training
1. `cd projects/train`
2. `apptainer build $DEEPCLEAN_CONTAINER_ROOT/train.sif apptainer.def`
3. Setting deepcleanv2/projects/train/config.yaml
    - To reduce the time of the test, "data:train_duration" and "data:test_duration" may be reduced.
4. `cd -`
5. Setting deepcleanv2/train.sh
    - Comment out the seeting of "DEEPCLEAN_CONTAINER_ROOT", "DATA_DIR", "RESULTS_DIR" if these environment variables are set in /export_vars.sh, otherwise we need to set them here.
    - We can choose the GPU to train the model by setting "GPU_INDEX".
    - Make sure "--data-fname", "train-config", "--output-dir" are set correctly.
6. `. train.sh`