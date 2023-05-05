#!/usr/bin/env bash

## run the test and export collapses
python3 test.py \
--gpu_ids -1 \
--dataroot ../dataset/sdata/ \
--name bcls \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--export_folder meshes \
