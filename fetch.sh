#!/bin/bash

poetry run law run deepclean.tasks.Fetch \
    --data-dir ${DATA_DIR} \
    --start 1378402219 \
    --end 1378402229 \
    --sample-rate 4096 \
    --min-duration 10 \
    --max-duration 32768 \
    --image data.sif \
    --job-log fetch.log
    # --end 1378405291 \
