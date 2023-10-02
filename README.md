# LungNoduleSegmentation
This is a project utilizing LIDC-IDRI open dataset to train both detection and segmentation models
## Introduction
Segmenting pulmonary nodules has always been a challenging task due to the small size of many target objects, making them difficult to separate. Additionally, some nodules located near the lung wall can also be affected in the edge segmentation process. Therefore, we aim to enhance accuracy in segmentation by implementing an attention mechanism.

## Method
### Preprocess
Since our dataset, LIDC-IDRI, consists of medical images from various medical centers, preprocessing is necessary to remove extremely high (e.g., bones, metal implants) or low (e.g., background) Hounsfield Unit (HU) values to eliminate unnecessary information that might interfere with segmentation. Due to hardware limitations, we utilized CT scans from a total of 250 patients for training. Furthermore, to address the class imbalance issue, we maintained a positive-to-negative sample ratio of 1:3 during training.
### Faster R-CNN
In order to improve the coverage of bounding boxes for pulmonary nodules in the Faster R-CNN model, we attempted the following modifications:
- Adjusted the initial sizes of generated candidate boxes.
- Modified the stride of Faster R-CNN ROI pooling.
- Changed the RPN (Region Proposal Network) classification loss function to BCE (Binary Cross-Entropy) + OHEM (Online Hard Example Mining).
- Altered the ROI classification loss function to weighted cross-entropy.
- Implemented the use of the multi-slice and patch approach to enable the model to learn information between slices.
- Applied post-processing techniques, including merging candidate boxes with an IoU (Intersection over Union) greater than 0.2.
![image](https://github.com/SamuelWu2001/LungNoduleSegmentation/assets/71746159/ab488c88-31cd-4088-8055-ae938329c59e)

### Unet
For the U-Net model, we introduced three cropping policies and compared their segmentation results. In addition, we incorporated attention modules into the skip connections and upsampling layers to assess whether their inclusion improved performance.
![image](https://github.com/SamuelWu2001/LungNoduleSegmentation/assets/71746159/bbe742e2-b67b-42e3-9ef7-0dc5df695642)


## Result
### Faster R-CNN
- Coverage: The average proportion of the area of each true bounding box that is covered by predicted bounding boxes.
- Recall: The proportion of all true bounding boxes for which Coverage exceeds 0.7.
- Coverage (true pos): The average coverage for all cases where Coverage exceeds 0.7 (true positives).
- Precision: The proportion of all predicted bounding boxes for which the Intersection over Union (IoU) with a true bounding box exceeds 0.3.
- IoU Average: The average IoU between all true bounding boxes and predicted bounding boxes.
- FP/Scan: The average number of false-positive predictions per slice.

| | Coverage | Recall | Coverage (true pos) | Precision | IoU Average | FP/Scan |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| With  Post process |  0.650 | 0.383 | 0.968 | 0.383(116/303) | 0.286 | 0.443 |
| Without  Post process | 0.631 | 0.373 | 0.953 | 0.516(353/684) | 0.382 | 0.784 |

### Unet
![image](https://github.com/SamuelWu2001/LungNoduleSegmentation/assets/71746159/7094b54c-3325-45ef-868a-80b80b960821)

### Attention Unet
| | IoU | Precision | Recall | Dice Score |
| -------- | -------- | -------- | -------- | -------- |
|  1    |  0.8235     | 0.9132     |0.8934     |0.8975   |
|  2   | 0.8021  |0.8879   |0.9104    |0.8913     |
|  3| 0.4711    | 0.7192    |0.6482 |0.6498    |
|   4   | 0.4329     |0.6522   |0.5115    |0.5567    |
|   5   | 0.2995     | 0.8373    |0.3131  |0.3947  |

method 1 : Utilizing accurate bounding box without padding for training and testing

method 2 : Utilizing accurate bounding box with padding for training and testing

method 3 : Training and testing without bounding box

method 4 : Utilizing faster R-CNN’s bounding box for testing in conjunction with method 1’s training model

method 5 : Utilizing faster R-CNN’s bounding box with padding for testing in conjunction with method 2’s training model

# Implementation
## Preprocess
- download the LIDC-IDRI open dataset and put the data under the folder named 'LIDC-IDRI'
- run config_file_create.py to create configure file
- run pip install -r requirements.txt to install necessary packages
- run prepare_dataset.py to obatain preprocessed data or run prepare_dataset_continuous.py to obatain continuous one
- run mask_to_bbox.ipynb to obtain bounding boxes

## Faster R-CNN
- custom rpn classification loss and roi classification loss can be fullfilled through removing comments
- For more modifications, please refer to the pytorch official website
- After training, run predict-fasterrcnn.ipynb to predict results

## Multi-slice Faster R-CNN
- Concatenate three consecutive slices and divide them into overlapping patches to serve as input for Faster R-CNN. The objective is to output the bounding box of lung nodules in the middle slice (the second slice).
- This method aims to test whether the information between multiple slices can assist in detection during the training phase. Therefore, only the input and output parts of the model have been implemented, excluding post-processing.

## UNet
- 3 cropping methods
  - origin: bounding box unchanged 
  - add padding: expand bounding box by certain pixels
  - none: use the entire image
- train with accurate bounding box information and test with FasterRCNN output

# Reference
[U-Net-based Models for Skin Lesion Segmentation: More Attention and Augmentation](https://arxiv.org/abs/2210.16399)
 
