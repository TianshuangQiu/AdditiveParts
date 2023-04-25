# AdditiveParts
UC Berkeley EECS282 Project

## Overview
This project has 4 components, each finished by a different team member. The parts are: baseline, 3D CNN, Mesh CNN, and pointcloud encoder. For the baseline and pointcloud computations, the instructions are outlined in this document. For 3D CNN and Mesh CNN, their respective README's are in their folders.

## Baseline
### To Install Original Printability Scores:
1. Download the .json file from the following link:https://drive.google.com/drive/folders/1ZDbtCF1YcgrH9zyQXbL-eH95N4o1lIvK?usp=sharing
### To Run
1. Change filepath at line 7 to filepath of downloaded .json file
2. Change filepath at line 10 to target filepath of result file
3. ```python run_baseline.py```

## Point Cloud Encoder
1. Download the zipped files here:
2. The file structure should be /base/rotated files, and /base/tweaker_score
3. Run utils/tensormaker.py with the proper arguments. This will go through the meshes and sample points and extract norms
4. Run utils/train.py with the proper arguments are hyperparameters. Please change the API key so you do not accidentally commit your run to our report. 
