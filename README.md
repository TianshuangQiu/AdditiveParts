# AdditiveParts
UC Berkeley EECS282 Project

## Overview
This project has 4 components, each finished by a different team member. The parts are: baseline, 3D CNN, Mesh CNN, and pointcloud encoder. For the baseline, 3D CNN, and pointcloud computations, the instructions are outlined in this document. For 3D CNN and Mesh CNN, their respective README's are in their folders.

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

## 3D CNN
### Downloading the Data
Meshes & Labels: https://drive.google.com/file/d/1Ny_H5a0CobkbChiAQMkfG6TdyDIMtKsd/view?usp=sharing

### Running the Network
1. Upload the 3D CNN notebook files (one_model_regression_once_data_loaded.ipynb, one_model_classification_once_data_loaded.ipynb, and one_model_classification_once_data_loaded_scale.ipynb) to the Google Drive repository you intend to use.
2. Download the meshes & labels file (link above) and move it to the Google Drive repository you intend to use. Change file path in "Load numpy array" block of code to match where you put the meshes & labels files.
3. Run any of the 3D CNN notebooks!
4. Note: you will need a wandB account for the wandB blocks of code to run. If you don't have a wandB account remove these sections of code (Wandb Install, Wandb Imports, Wandb Ininitialization, Finish WandB, and the callbacks in Training Model and Evaluate Model) and the code should still run.

## Point Cloud Encoder
### Downloading the Data:
Meshes: https://drive.google.com/drive/folders/1C0MGixYalkqlBkXeAsyGjXVHUfst113t?usp=share_link

Labels: https://drive.google.com/drive/folders/1XQ1MZiSwdev-85kfswiY5qveu_y-cWvt?usp=sharing
### Running the Network
1. Unzip the file and clone the repositories
2. The file structure should be `/base/rotated_files`, `/base/tweaker_score`, and `base/AdditiveParts`
3. Run utils/tensormaker.py specifying the path to your base folder, the rotated_files folder, and the folder for CSVs. This script will go through the meshes and sample points and extract norms
4. Run utils/train.py with the proper arguments and hyperparameters. Please change the API key so you do not accidentally commit your run to our report.


## MeshCNN
### Downloading the Data:
Dataset: [https://drive.google.com/drive/folders/1C0MGixYalkqlBkXeAsyGjXVHUfst113t?](https://drive.google.com/file/d/1yejuhFTWGWXTpzZ6iesxitI513-e14hR/view?usp=sharing)
### Running the Network
1. Unzip the dataset. Make sure the unzipped folder is named "sdata"
2. Move the dataset into the ./MeshCNN/dataset/
3. The filepath to the data should look like ./MeshCNN/dataset/sdata
4. from the root folder of the repo, run `cd ./MeshCNN/MeshCNN/`
6. Run `bash scripts/keene/test.sh`
