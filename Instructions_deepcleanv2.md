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
    - Setting subtraction problem, the problems can be found in /deepclean/couplings.
3. `. export_vars.sh`
4. Setting deepcleanv2/luigi.cfg
    - Setting container_root.
5. `cd projects/data`
6. `apptainer build $DEEPCLEAN_CONTAINER_ROOT/data.sif apptainer.def`
7. `cd -`
8. Setting deepcleanv2/fetch.sh
9. `. fetch.sh`

## Training
1. `cd projects/train`
2. `apptainer build $DEEPCLEAN_CONTAINER_ROOT/train.sif apptainer.def`
3. Setting deepcleanv2/projects/train/config.yaml
    - To reduce the time of the test, "data:train_duration" and "data:test_duration" may be reduced.
4. `cd -`
5. Setting deepcleanv2/train.sh
    - Comment out the setting of "DEEPCLEAN_CONTAINER_ROOT", "DATA_DIR", "RESULTS_DIR" if these environment variables are set in /export_vars.sh, otherwise we need to set them here.
    - We can choose the GPU to train the model by setting "GPU_INDEX".
    - Make sure "--data-fname", "train-config", "--output-dir" are set correctly.
6. `. train.sh`

## Data Replay
1. `mkdir -p dc-review/ll_data/kafka/H1`
2. `mkdir -p dc-review/ll_data/lldetchar/H1`
3. `mkdir -p dc-review/ll_data/llhoft_buffer/H1`
4. `mkdir -p dc-review/ll_data/lldetchar_buffer/H1`
5. `mkdir -p dc-review/ll_data/unresampled_data/H1_HOFT`
6. `mkdir -p dc-review/ll_data/unresampled_data/H1_INMON`
7. `cd dc-review/deepcleanv2/data_replay/get_lldata`
8. Setting get_data.sh to get llhoft data by setting "KIND=llhoft".
9. `bash get_data.sh`
10. Setting get_data.sh to get llhoft data by setting "KIND=lldetchar".
11. `bash get_data.sh`
12. `cd dc-review/deepcleanv2/data_replay/make_lldata`
13. Setting make_lldata.sh to make 1-second long gwf files in the llhoft_buffer folder by setting "KIND=llhoft".
14. `bash make_lldata.sh`
15. Setting make_lldata.sh to make 1-second long gwf files in the lldetchar_buffer folder by setting "KIND=lldetchar".
16. `bash make_lldata.sh`
17. `cd dc-review/deepcleanv2/data_replay/replay`
18. Setting start_replay.sh so the 1-second log gwf files in the buffer folders can be copied to the kafka folder and lldetchar folder.
19. `bash start_replay.sh`
20. To stop the replay, just run `bash stop_replay.sh`

## Cleaning
1. Setting deepcleanv2/projects/clean/config_clean.yaml
2. `cd projects/clean`
2. `poetry install`
3. `bash run-clean.sh`
