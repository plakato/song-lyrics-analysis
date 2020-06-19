import os
import shutil

import numpy as np
from shutil import copy2


# Load data and split to test/train.

# PARAMETERS
BASE_PATH = '../sparsar_experiments/repetition_matrices'
DATASET_PATH = 'dataset'
TRAIN_RATIO = 0.8
PREFIX = 'endrhymes_'

# Clear the directories to get rid of previous data.
if os.path.exists(DATASET_PATH) and os.path.isdir(DATASET_PATH):
    shutil.rmtree(DATASET_PATH)

# Specify path to the downloaded folder
classes = os.listdir(BASE_PATH)

# Specify path for copying the dataset into train and val sets
os.makedirs(DATASET_PATH, exist_ok=True)

# Creating train directory
train_dir = os.path.join(DATASET_PATH, 'train')
os.makedirs(train_dir, exist_ok=True)

# Creating val directory
val_dir = os.path.join(DATASET_PATH, 'val')
os.makedirs(val_dir, exist_ok=True)

# Copying images from original folder to dataset folder
for class_name in classes:
    if len(class_name.split('.')) >= 2:
        continue
    clean_class_name = class_name.replace(PREFIX, '')
    print(f"Copying images for {clean_class_name}...")

    # Creating destination folder (train and val)
    class_train_dir = os.path.join(train_dir, clean_class_name)
    os.makedirs(class_train_dir, exist_ok=True)

    class_val_dir = os.path.join(val_dir, clean_class_name)
    os.makedirs(class_val_dir, exist_ok=True)

    # Shuffling the image list
    class_path = os.path.join(BASE_PATH, class_name)
    class_images = os.listdir(class_path)
    np.random.shuffle(class_images)

    train_images = int(len(class_images)*TRAIN_RATIO)
    for image in class_images[:train_images]:
        copy2(os.path.join(class_path, image), class_train_dir)
    for image in class_images[train_images:]:
        copy2(os.path.join(class_path, image), class_val_dir)