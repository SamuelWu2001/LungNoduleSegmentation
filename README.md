# LungNoduleSegmentation
It is a project which utilizing LIDC-IDRI open dataset to train both detection and segmentation models

## Preprocess
- download the LIDC-IDRI open dataset and put the data under the folder named 'LIDC-IDRI'
- run config_file_create.py to create configure file
- run pip install -r requirements.txt to install necessary package
- run prepare_dataset.py to obatain preprocessed data 
- run prepare_dataset_continuous.py to obatain continuous preprocessed data 
- run mask_to_bbox.ipynb to obtain bounding boxes
