#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--gpu_ids -1 \
--dataroot datasets/shrec_16 \
--name shrec16 \
--ncf 64 128 512 \
--pool_res 450 300 180 \
--norm group \
--resblocks 1 \
--export_folder meshes \
