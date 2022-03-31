# Robust Category-Domain Classification Using Causal Deep Learning Representations

### Bennington Li, James Wang, Xiao Yu
#### COMS 4995 Deep Learning Final Project
---
## Setup

* Create the environment from `environment.yml` file and `requirements.txt`. 
* Run `conda env create -f environment.yml` and `conda activate envName`
* Run `pip install requirements.txt`

## Project Structure

```bash
root
├── datasets/ 		# contains the datasets we used
├── logs/ 			# logging for models
├── models/ 		# model implementations
├── preprocess/ 	# data preprocessing
├── *.ipynb 		# experimental code
└── *.py
```

## Data Preprocessing
Either
* Download the OfficeHome and PACS datasets from online.
* Run the `preprocess/*.ipynb` Jupyter notebooks to preprocess the data for training and testing object category and domain. 

Or
- due to large size, you can find train/test splits used directly at https://drive.google.com/drive/folders/1f7m3-aYCLIgNmZoi_aRFHTEAzEq5K90-?usp=sharing


## Model Training - Benchmarks

### Alexnet - Shared

- run `python models/alexnet_shared.py` for classifying domain and object at the same time using a shared backbone on PACS dataset. 

### Alexnet - Separate

- Run `python models/1a_Train_TL_Resnet18_Separate_PACS.py` for domain classification on PACS dataset. 
- Run `python models/1b_Train_TL_Resnet18_Separate_PACS.py` for object classification on PACS dataset. 

### Alexnet - Shared, Pretrained

[TODO]

### Alexnet - Separate, Pretrained

[TODO]


## Model Training - Causal Representations

[TODO]