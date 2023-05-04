# AdditiveParts
UC Berkeley EECS282 Project

## Overview
This project has 4 components, each finished by a different team member. The parts are: baseline, 3D CNN, Mesh CNN, and pointcloud encoder. For the baseline, 3D CNN, and pointcloud computations, the instructions are outlined in this document. For 3D CNN and Mesh CNN, their respective README's are in their folders.

Additional hyperparameter tuning and discussion can be found at the following links:

https://wandb.ai/additive-parts/additive-parts/reports/Point-Cloud-Encoder-Regression---Vmlldzo0MTQ5MTU4

https://wandb.ai/additive-parts/additive-parts/reports/Point-Cloud-Encoder-Classification---Vmlldzo0MTY3OTA1


## File Structure
Please download the files from this folder: https://drive.google.com/drive/folders/1jCzhg4bwJk7lqaEBTcfoWZch0BIdxavM?usp=sharing and arrange into the following structure
```
BASE_DIR
├── AdditiveParts
├── rawcloud
├── rawcloud.json
├── rawnorm
├── rawnorm.json
├── repaired_files
```


## Baseline
1. Change filepath at line 7 to filepath of downloaded .json file
2. Change filepath at line 10 to target filepath of result file
3. ```python run_baseline.py```

## Point Cloud Encoder
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
