#!/usr/bin/env bash

CONFIG=a_configs/paper3_OPA/voc/GA/ga_r101_voc.py

GPUS=8
PORT=${PORT:-298}
find_unused_parameters=True
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
