   #!/usr/bin/env bash

## run the training
python train.py \
--cleanup_mode 1 \
--gpu_ids -1 \
--dataroot ../dataset/sdata/ \
--ncf 64 128 512 \
--pool_res 450 300 180 \
--norm group \
--batch_size 1 \
