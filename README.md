# AdditiveParts
UC Berkeley EECS282 Project

## Overview
This project has 4 components, each finished by a different team member. The parts are: baseline, 3D CNN, Mesh CNN, and pointcloud encoder. For the baseline and pointcloud computations, the instructions are outlined in this document. For 3D CNN and Mesh CNN, their respective README's are in their folders.

Additional hyperparameter tuning and discussion can be found at the following links:

https://wandb.ai/additive-parts/additive-parts/reports/Point-Cloud-Encoder-Regression---Vmlldzo0MTQ5MTU4

https://wandb.ai/additive-parts/additive-parts/reports/Point-Cloud-Encoder-Classification---Vmlldzo0MTY3OTA1


## Baseline
### To Install Original Data and Printability Scores:
Download the dataset from the following links:

Meshes: https://drive.google.com/drive/folders/1C0MGixYalkqlBkXeAsyGjXVHUfst113t?usp=share_link

Labels: https://drive.google.com/drive/folders/1XQ1MZiSwdev-85kfswiY5qveu_y-cWvt?usp=sharing
### To Run
1. Unzip the files and clone the repositories
2. The file structure should be `/base/rotated_files`, `/base/tweaker_score`, and `base/AdditiveParts`
3. Run utils/tensormaker.py specifying the path to your base folder, the rotated_files folder, and the folder for CSVs. This script will extract the stl filepaths and their corresponding scores into a json file.
4. In `run_baseline.py`, change filepath at line 7 to filepath of created json file, also change filepath at line 10 to target filepath of resulting json file
6. ```python run_baseline.py```

## Point Cloud Encoder
### Downloading the Data:
Meshes: https://drive.google.com/drive/folders/1C0MGixYalkqlBkXeAsyGjXVHUfst113t?usp=share_link

Labels: https://drive.google.com/drive/folders/1XQ1MZiSwdev-85kfswiY5qveu_y-cWvt?usp=sharing
### Running the Network
1. Unzip the file and clone the repositories
2. The file structure should be `/base/rotated_files`, `/base/tweaker_score`, and `base/AdditiveParts`
3. Run utils/tensormaker.py specifying the path to your base folder, the rotated_files folder, and the folder for CSVs. This script will go through the meshes and sample points and extract norms
4. Run utils/train.py with the proper arguments and hyperparameters. Please change the API key so you do not accidentally commit your run to our report. 
