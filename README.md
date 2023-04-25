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
### Downloading the Data:
Meshes: https://drive.google.com/drive/folders/1C0MGixYalkqlBkXeAsyGjXVHUfst113t?usp=share_link
Labels: https://drive.google.com/drive/folders/1XQ1MZiSwdev-85kfswiY5qveu_y-cWvt?usp=sharing
### Running the Network
1. Unzip the file and clone the repositories
2. The file structure should be `/base/rotated_files`, `/base/tweaker_score`, and `base/AdditiveParts`
3. Run utils/tensormaker.py specifying the path to your base folder, the rotated_files folder, and the folder for CSVs. This script will go through the meshes and sample points and extract norms
4. Run utils/train.py with the proper arguments and hyperparameters. Please change the API key so you do not accidentally commit your run to our report. 
