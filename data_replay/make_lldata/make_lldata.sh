#!/bin/bash

IFO=H1
HOFT_TAG=HOFT
DETCHAR_TAG=INMON
HOFT_SOURCE=/home/chiajui.chou/ll_data/unresampled_data/${IFO}_${HOFT_TAG}
DETCHAR_SOURCE=/home/chiajui.chou/ll_data/unresampled_data/${IFO}_${DETCHAR_TAG}
HOFT_DESTINATION=/home/chiajui.chou/ll_data/llhoft_buffer/${IFO}
DETCHAR_DESTINATION=/home/chiajui.chou/ll_data/lldetchar_buffer/${IFO}
START=1250916945
DURATION=12288
KIND=lldetchar

python make_lldata.py \
    --ifo ${IFO} \
    --hoft_source ${HOFT_SOURCE} \
    --detchar_source ${DETCHAR_SOURCE} \
    --hoft_destination ${HOFT_DESTINATION} \
    --detchar_destination ${DETCHAR_DESTINATION} \
    --start ${START}\
    --duration ${DURATION}\
    --kind ${KIND}