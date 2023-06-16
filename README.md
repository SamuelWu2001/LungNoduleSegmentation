# LungNoduleSegmentation
It is a project which utilizing LIDC-IDRI open dataset to train both detection and segmentation models

## Preprocess
- download the LIDC-IDRI open dataset and put the data under the folder named 'LIDC-IDRI'
- run config_file_create.py to create configure file
- run pip install -r requirements.txt to install necessary packages
- run prepare_dataset.py to obatain preprocessed data or run prepare_dataset_continuous.py to obatain continuous one
- run mask_to_bbox.ipynb to obtain bounding boxes

## FasterRCNN
- custom rpn classification loss and roi classification loss can be fullfilled through removing comments
- For more modifications, please refer to the pytorch official website
- After training, run predict-fasterrcnn.ipynb to predict results
