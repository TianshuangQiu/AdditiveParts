# AdditiveParts
UC Berkeley EECS282 Project

## Overview
This project has 4 components, each finished by a different team member. The parts are: baseline, 3D CNN, Mesh CNN, and pointcloud encoder. We have streamlined some portions of our process for ease of replicating our results, to test our code for yourself, please check the `run` folder.

Additional hyperparameter tuning and discussion can be found at the following links:

[PCE Regression](https://wandb.ai/additive-parts/synced-parts/reports/Point-Cloud-Encoder-Regression---Vmlldzo0MjY0NDM2?accessToken=xp6jexcql35n2jwbbti9fgutwf1opr130xj5nwka3135wgjhwsb3tbz3362votuq)

[PCE Classification](https://wandb.ai/additive-parts/synced-parts/reports/Point-Cloud-Encoder-Classification---Vmlldzo0MjY0MTY4?accessToken=krxx4xcdwttrzf1sz7fdefzc0w4rjf29uug0c52godobod1304w3rj5sh9tmmgun)

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
### Stable Pose
1. Change filepath at line 7 to filepath of downloaded .json file
2. Change filepath at line 10 to target filepath of result file
3. ```python run_baseline.py```
### Optimized Normal
1. Run the `run/normalBaseline.ipynb` notebook

## Point Cloud Encoder
1. Run the `run/trainPCE.ipynb` notebook, it walks through how to run our experiments.


## MeshCNN
### Downloading the Data:
Dataset: [https://drive.google.com/drive/folders/1C0MGixYalkqlBkXeAsyGjXVHUfst113t?](https://drive.google.com/file/d/1yejuhFTWGWXTpzZ6iesxitI513-e14hR/view?usp=sharing)
### Running the Network (cd into the AdditiveParts folder first)
1. Unzip the dataset. Make sure the unzipped folder is named "sdata"
2. Move the dataset into the ./meshcnn/dataset/
3. The filepath to the data should look like ./meshcnn/dataset/sdata
4. from the root folder of the repo do `cd ./meshcnn/MeshCNN/`
5. Create a virtual environment using `conda create -n meshcnn python=3.6.8`, activate the virtual environment and run `pip install -r requirements.txt` inside.  
6. Run `bash scripts/keene/test.sh` to test
