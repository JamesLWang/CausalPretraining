# Robust Category-Domain Classification Using Causal Deep Learning Representations

### Bennington Li, James Wang, Xiao Yu
#### COMS 4995 Deep Learning Final Project
---
## Setup

* Create the environment from `environment.yml` file and `requirements.txt`. 
* Run `conda env create -f environment.yml` and `conda activate envName`
* Run `pip install requirements.txt`
---

## Data Preprocessing

* Download the OfficeHome and PACS datasets from online.
* Run the `0_DataProcessing_OfficeHome` and `0_DataProcessing_PACS` Jupyter notebooks to preprocess the data for training and testing object category and domain. 
---

## Model Training - Benchmarks

### Alexnet - Shared

[TODO]

### Alexnet - Separate

* Run `python 1a_Train_TL_Resnet18_Separate_PACS.py` for domain classification and `python 1b_Train_TL_Resnet18_Separate_PACS.py` for object classification on PACS dataset. 

### Alexnet - Shared, Pretrained

[TODO]

### Alexnet - Separate, Pretrained

[TODO]
---

## Model Training - Causal Representations

[TODO]