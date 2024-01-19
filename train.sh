# name num device epoch lr batch-size attn-repeat num-layer
export CUDA_VISIBLE_DEVICES=3
python3 scripts/trainPCE.py capped_10k16repeat 10000 10 0.01 128 16 8
python3 scripts/trainPCE.py capped_10k8repeat 10000 10 0.01 128 8 8
python3 scripts/trainPCE.py capped_50k16repeat 50000 10 0.01 128 16 8
python3 scripts/trainPCE.py capped_50k8repeat 50000 10 0.01 128 8 8
python3 scripts/trainPCE.py capped_100k16repeat 100000 10 0.01 128 16 8
python3 scripts/trainPCE.py capped_100k8repeat 100000 10 0.01 128 8 8
python3 scripts/trainPCE.py capped_200k16repeat 200000 10 0.01 128 16 8
python3 scripts/trainPCE.py capped_200k8repeat 200000 10 0.01 128 8 8
