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
## Model Architecture
For category classification:
![Category Classification Architecture](images/archi.png)
where essentially:
- a Resnet-18 is used for modelling $P(Z|X)$
- the final classifier models on $P(Y | Z, X)$
- the sampling time for computing sums is not shown here

For domain classification essentially the final classifier outputs a 4 dimensional vector, representing the 4 domains in the PACS dataset.

## Data Preprocessing
Either
* Download the OfficeHome and PACS datasets from online.
* Run the `preprocess/*.ipynb` Jupyter notebooks to preprocess the data for training and testing object category and domain. 

Or
- due to large size, you can find train/test splits used directly at https://drive.google.com/drive/folders/1f7m3-aYCLIgNmZoi_aRFHTEAzEq5K90-?usp=sharing


## Model Training - Benchmarks

### Resnet-18 Causal - In-Distribution

 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_sketch_id --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_photo_id --style P`
 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_art_id --style A`
 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_cartoon_id --style C`
 - run `CUDA_VISIBLE_DEVICES=0 python 3b_fd_pacs_cat_ood.py --fast --train_all --drop_xp --fname pacs_sketch_id --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 3c_fd_eval.py`
 - run `CUDA_VISIBLE_DEVICES=0 python 3c_fd_evalCat.py`
 
### Resnet-18 Causal - Out-of-Distribution

Same as before, but change the training path to the out-of-distribution one from preprocessing PACS dataset. 

 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_sketch_ood --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_photo_ood --style P`
 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_art_ood --style A`
 - run `CUDA_VISIBLE_DEVICES=0 python 3a_fd_pacs.py --fast --train_all --drop_xp --fname pacs_cartoon_ood --style C`
 - run `CUDA_VISIBLE_DEVICES=0 python 3b_fd_pacs_cat_ood.py --fast --train_all --drop_xp --fname pacs_sketch_ood --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 3c_fd_eval.py`
 - run `CUDA_VISIBLE_DEVICES=0 python 3c_fd_evalCat.py`
 
### Resnet-18 Baseline - Out-of-Distribution

 - run `CUDA_VISIBLE_DEVICES=0 python 4a_fd_pacs_ood.py --fast --train_all --drop_xp --fname pacs_sketch_ood_bl --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 4a_fd_pacs_ood.py --fast --train_all --drop_xp --fname pacs_photo_ood_bl --style P`
 - run `CUDA_VISIBLE_DEVICES=0 python 4a_fd_pacs_ood.py --fast --train_all --drop_xp --fname pacs_art_ood_bl --style A`
 - run `CUDA_VISIBLE_DEVICES=0 python 4a_fd_pacs_ood.py --fast --train_all --drop_xp --fname pacs_cartoon_ood_bl --style C`
 - run `CUDA_VISIBLE_DEVICES=0 python 4b_fd_pacs_cat_ood.py --fast --train_all --drop_xp --fname pacs_sketch_ood_bl --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 4c_fd_eval_ood.py`
 
### Resnet-18 Baseline - In-Distribution

 - run `CUDA_VISIBLE_DEVICES=0 python 5a_fd_pacs_resnet18.py --fast --train_all --drop_xp --fname pacs_sketch_id_bl --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 5a_fd_pacs_resnet18.py --fast --train_all --drop_xp --fname pacs_photo_id_bl --style P`
 - run `CUDA_VISIBLE_DEVICES=0 python 5a_fd_pacs_resnet18.py --fast --train_all --drop_xp --fname pacs_art_id_bl --style A`
 - run `CUDA_VISIBLE_DEVICES=0 python 5a_fd_pacs_resnet18.py --fast --train_all --drop_xp --fname pacs_cartoon_id_bl --style C`
 - run `CUDA_VISIBLE_DEVICES=0 python 5b_fd_pacs_cat_resnet18.py --fast --train_all --drop_xp --fname pacs_sketch_id_bl --style S`
 - run `CUDA_VISIBLE_DEVICES=0 python 5c_fd_eval_resnet.py`
