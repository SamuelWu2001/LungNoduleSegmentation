# LungNoduleSegmentation
It is a project utilizing LIDC-IDRI open dataset to train both detection and segmentation models

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

## Multi-slice FasterRCNN
- Concatenate three consecutive slices and divide them into overlapping patches to serve as input for Faster R-CNN. The objective is to output the bounding box of lung nodules in the middle slice (the second slice).
- This method aims to test whether the information between multiple slices can assist in detection during the training phase. Therefore, only the input and output parts of the model have been implemented, excluding post-processing.

## UNet
- 3 cropping methods
  - origin: bounding box unchanged 
  - add padding: expand bounding box by certain pixels
  - none: use the entire image
- train with accurate bounding box information and test with FasterRCNN output
 
