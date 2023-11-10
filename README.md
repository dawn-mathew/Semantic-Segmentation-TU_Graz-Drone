# Semantic-Segmentation-TU_Graz-Drone
## Introduction
This repository offers a basic & fundamental implementation for testing and training a neural network on the TU drone dataset (ICG)or semantic segmentation. It includes the dataset with annotations and basic code to evaluate performance and accuracy. This resource is designed for initial experimentation and exploration of semantic segmentation.
> Note: The dataset provided here is the complete one due to file size restrictions. Downnload the original dataset from [here](https://www.tugraz.at/index.php?id=22387).

## Repo Structure
```
├── class_dict_seg.csv
├── dataset
│   └── semantic_drone_dataset
│       ├── label_images_semantic
│       └── original_images
├── resnet_backbone.hdf5
├── RGB_color_image_masks
│   └── RGB_color_image_masks
└── Semantic_Segmentation.ipynb

```
- `class_dict_seg.csv` is a csv file containing a class dictionary or mapping for semantic segmentation, associating class labels with their corresponding semantic meanings.
- `label_images_semantic` is a subdirectory containing labeled images for semantic segmentation. These are gray scale masks that can be used to train the neural network for the semantic segmentation task.
- `original_images` is a subdirectory where the original drone images are stored, which are used for semantic segmentation training and testing.
- 'RGB_color_image_masks' is the subdirectory that contains RGB color masks, used for training the neural network for semantic segmentation.
- `Semantic_Segmentation.ipynb` is a Jupyter Notebook containing the code and documentation related to performing semantic segmentation tasks on the provided dataset.

## Dependencies
Install all of the following libraries. Make sure that in the virtual environment `os.environ["SM_FRAMEWORK"] = "tf.keras"` is ran prior to the main code after importing `tensorflow `
```py
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tqdm import tqdm
import random
import pickle
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,  EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
```
## Results
- During the training of our neural network over 100 epochs, with a batch size of 16, we observed significant advancements. The network achieved an impressive accuracy of 0.9075 and a final loss of 0.2965 on the training dataset, indicating its strong ability to capture underlying patterns. Meanwhile, on the validation set, it displayed notable performance, with a loss of 1.4093 and an accuracy of 0.7158. These results underscore the network's capacity to generalize learned knowledge to unseen data, underscoring its effectiveness in our training task.

![image](https://github.com/dawn-mathew/Semantic-Segmentation-TU_Graz-Drone/assets/150279674/47832464-3c56-4497-875f-828a1b57a9ff)
- The model's ability to perform semantic segmentation on new data is evident from the images bellow. With reasonable accuracy, it effectively assigns semantic labels to objects within the images, demonstrating its aptitude for generalization.
![image](https://github.com/dawn-mathew/Semantic-Segmentation-TU_Graz-Drone/assets/150279674/e4545d99-e95f-49f3-825e-b9c7b7502dfe)


