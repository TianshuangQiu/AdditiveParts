#!/bin/bash

cd scripts/
bash process.sh
cd ../Manifold/build
bash ../../scripts/manifold.sh
cd ../../
python scripts/dataify.py
cd MeshCNN
bash scripts/keene/train.sh
