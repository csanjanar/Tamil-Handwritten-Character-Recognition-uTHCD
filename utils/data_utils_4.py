import numpy as np
import random
import math
import h5py
import matplotlib.pyplot as plt
# Tensorflow 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# ImageDataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Albumentations
import albumentations as A
from tqdm import tqdm
# OpenCV
import cv2
# sklearn
from sklearn.utils import shuffle


# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def load_data(file_path):
    """
    Load and preprocess data from an HDF5 file.

    Parameters:
    - file_path: str, path to the HDF5 file

    Returns:
    - tuple: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    # Open the uTHCD compressed HDF5 file in read mode
    with h5py.File(file_path, 'r') as hdf:
        # Extract training data and labels and convert to numpy arrays
        x_train_full = np.array(hdf['Train Data']['x_train'])
        y_train_full = np.array(hdf['Train Data']['y_train'])
        
        # Extract test data and labels and convert to numpy arrays
        x_test = np.array(hdf['Test Data']['x_test'])
        y_test = np.array(hdf['Test Data']['y_test'])
        
        # Extract validation set from the training set (last 7870 samples - 1/8th of Train Data)
        x_val = x_train_full[-7870:]
        y_val = y_train_full[-7870:]
        
        # Update training set to exclude the validation set
        x_train = x_train_full[:-7870]
        y_train = y_train_full[:-7870]

        val_size = x_val.shape[0]

    # Print shapes to verify the extracted data
    print(f"Data loaded ==============================")
    print(f"Extracted data (X) and labels (Y) shapes:")
    print(f"Train Set| X: {x_train_full.shape}, Y: {y_train_full.shape}")
    print(f"Test Set | X: {x_test.shape},  Y: {y_test.shape}")
    print(f"Train-Validation Split ===================")
    print(f"Training data    | X: {x_train.shape}, Y: {y_train.shape}")
    print(f"Validation data  | X: {x_val.shape},   Y: {y_val.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), val_size 

def view_sample_images(images, labels, num_rows, title):
    """
    Display sample images with integer labels.
    
    Parameters:
    - images: numpy array, image data
    - labels: numpy array, one-hot encoded labels or integer labels
    - num_rows: int, number of rows of images to display
    - title: str, title of the plot
    """
    # Decode one-hot encoded labels to integers if necessary
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    # Determine the number of samples to display
    num_samples = num_rows * 8
    
    # Create a figure to display the images
    plt.figure(figsize=(16, num_rows*2 + num_rows))
    
    for i in range(num_samples):
        # Select a random index
        idx = np.random.randint(0, len(images))
        # Create a subplot
        plt.subplot(num_rows, 8, i + 1)
        # Display the image and use squeeze to handle single channel images
        plt.imshow(images[idx].squeeze(), cmap='gray')
        # Display the label as the title
        plt.title(f'Label: {labels[idx]}')
        # Hide the axis
        plt.axis('off')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.suptitle(f"{title}", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

def normalize_data(x_train, x_val, x_test):
    """
    Normalize the image data to the range [0, 1] by dividing by 255.0.

    Parameters:
    - x_train: numpy array, training images
    - x_val: numpy array, validation images
    - x_test: numpy array, test images

    Returns:
    - tuple: normalized training, validation, and test images
    """
    # Normalize the data by dividing by 255.0
    return x_train / 255.0, x_val / 255.0, x_test / 255.0

def encode_labels(y_train, y_val, y_test):
    """
    One-hot encode the labels for training, validation, and test datasets.

    Parameters:
    - y_train: numpy array, training labels
    - y_val: numpy array, validation labels
    - y_test: numpy array, test labels

    Returns:
    - list: one-hot encoded training, validation, and test labels
    """
    # Determine the number of unique classes in the training labels
    num_classes = len(np.unique(y_train))
    
    # Verify the number of unique classes
    print(f"Number of unique classes: {num_classes}")
    
    # One-hot encode the labels for each dataset
    return [to_categorical(y, num_classes) for y in (y_train, y_val, y_test)]

def ensure_4d(data):
    """
    Ensure the input data is 4-dimensional, adding a channel dimension if necessary.

    Parameters:
    - data: numpy array, input data

    Returns:
    - numpy array: reshaped data with 4 dimensions
    """
    # Check if the data is not already 4-dimensional
    if data.ndim != 4:
        # Reshape the data to add the channel dimension (for grayscale images)
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    elif data.ndim == 4:
        # If the data is already 4-dimensional, return it as is
        return data
    else:
        # Raise an error if the data has an unexpected number of dimensions
        raise ValueError(f"Unexpected number of dimensions: {data.ndim}.")
    
    return data

# # (2) With Morphological operations
# 
# def apply_morphological_ops(image, **kwargs):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     if random.randint(1, 5) == 1:
#         # Erosion (because the image is not inverted)
#         image = cv2.erode(image, kernel, iterations=random.randint(1, 2))
#     if random.randint(1, 6) == 1:
#         # Dilation (because the image is not inverted)
#         image = cv2.dilate(image, kernel, iterations=random.randint(1, 1))
#     return image

# def get_augmentation_pipeline():
#     # Define white color for borders (255 for grayscale images)
#     border_value = 255
#     return A.Compose([
#         A.RandomScale(scale_limit=0.1, p=0.7),
#         A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=border_value),
#         A.Affine(translate_percent={'x': (-0.1575, 0.1575), 'y': (-0.1575, 0.1575)}, p=0.7, 
#                  mode=cv2.BORDER_CONSTANT, cval=border_value),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5,
#                           border_mode=cv2.BORDER_CONSTANT, value=border_value),
#         A.Lambda(image=apply_morphological_ops, p=0.5),  # Add morphological operations
#         A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
#         A.Blur(blur_limit=3, p=0.2),
#         A.Resize(64, 64, always_apply=True)
#     ], p=1)

def apply_morphological_ops(image, **kwargs):
    # Create an elliptical kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1: # 20% chance of applying erosion
        # Erode the image 1-2 times
        image = cv2.erode(image, kernel, iterations=random.randint(1, 2))
    if random.randint(1, 6) == 1: # ~16.7% chance of applying dilation
        # Dilate the image once
        image = cv2.dilate(image, kernel, iterations=random.randint(1, 1))
    return image

def get_augmentation_pipeline():
    border_value = 255 # Define white color for borders (255 for grayscale images)
    return A.Compose([
        # Randomly scale the image up or down by up to 10%
        A.RandomScale(scale_limit=0.1, p=0.1),
        # Rotate the image by up to 15 degrees
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=border_value),
        A.OneOf([
            # Apply affine transformations (horizontal/vertical translation) 
            A.Affine(translate_percent={'x': (-0.1575, 0.1575), 'y': (-0.1575, 0.1575)}, p=0.1,
                 mode=cv2.BORDER_CONSTANT, cval=border_value),
            # Combination of shift, scale, and rotate
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3,
                           border_mode=cv2.BORDER_CONSTANT, value=border_value)
        ], p=0.5)
        # Apply custom morphological operations
        A.Lambda(image=apply_morphological_ops, p=0.5),
        # Add Gaussian noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # Adjust brightness and contrast
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # Apply blur
        A.Blur(blur_limit=3, p=0.2),
        # Ensure final image size is 64x64
        A.Resize(64, 64, always_apply=True)
    ], p=1)  # Apply the entire pipeline with 100% probability

def full_augmentation(x_train, y_train, augmentation_pipeline, augmentation_factor):
    total_samples = math.ceil(len(x_train) * augmentation_factor)
    x_augmented = np.zeros((total_samples, 64, 64), dtype=x_train.dtype)
    y_augmented = np.zeros(total_samples, dtype=y_train.dtype)
    
    for i in tqdm(range(total_samples), desc="Full Augmentation"):
        idx = i % len(x_train)
        image = x_train[idx].astype(np.uint8)  # Ensure image is uint8
        augmented = augmentation_pipeline(image=image)
        x_augmented[i] = augmented['image']
        y_augmented[i] = y_train[idx]
    
    return x_augmented, y_augmented

def random_augmentation(x_train, y_train, val_size, augmentation_pipeline, augmentation_factor):
    num_to_augment = math.ceil(val_size * augmentation_factor)
    rng = np.random.default_rng(42)
    indices_to_augment = rng.choice(len(x_train), num_to_augment, replace=True)
    
    x_augmented = np.zeros((num_to_augment, 64, 64), dtype=x_train.dtype)
    y_augmented = np.zeros(num_to_augment, dtype=y_train.dtype)
    
    for i, idx in enumerate(tqdm(indices_to_augment, desc="Random Augmentation")):
        image = x_train[idx].astype(np.uint8)  # Ensure image is uint8
        augmented = augmentation_pipeline(image=image)
        x_augmented[i] = augmented['image']
        y_augmented[i] = y_train[idx]
    
    return x_augmented, y_augmented

def augment_data(x_train, y_train, val_size, augmentation_factor):
    augmentation_pipeline = get_augmentation_pipeline()

    full_aug_x_train, full_aug_y_train = full_augmentation(x_train, y_train, augmentation_pipeline, augmentation_factor)
    random_aug_x_train, random_aug_y_train = random_augmentation(x_train, y_train, val_size, augmentation_pipeline, augmentation_factor)
    
    aug_x_train = np.concatenate((full_aug_x_train, random_aug_x_train), axis=0)
    aug_y_train = np.concatenate((full_aug_y_train, random_aug_y_train), axis=0)

    print(f"==> Total augmented samples (Factor {augmentation_factor}): {aug_x_train.shape[0]}")
    print(f"Total augmented data shapes | X: {aug_x_train.shape}, Y: {aug_y_train.shape}")
    
    return aug_x_train, aug_y_train

def prepare_data_for_training(x_train, y_train, x_val, y_val, x_test, y_test, aug_x_train=None, aug_y_train=None):
    """
    Prepare data for training by optionally combining original and augmented data, normalizing, and encoding labels.
    Parameters:
    - x_train_split: numpy array, original training images
    - y_train_split: numpy array, original training labels
    - x_val: numpy array, validation images
    - y_val: numpy array, validation labels
    - x_test: numpy array, test images
    - y_test: numpy array, test labels
    - aug_x_train: numpy array, augmented training images (optional)
    - aug_y_train: numpy array, augmented training labels (optional)
    Returns:
    - tuple: combined and preprocessed training, validation, and test datasets
    """
    # Ensure all data has the same shape (4D)
    x_train = ensure_4d(x_train)
    x_val = ensure_4d(x_val)
    x_test = ensure_4d(x_test)
    if aug_x_train is not None:
        aug_x_train = ensure_4d(aug_x_train)
    
    # Combine original and augmented training data if augmented data is provided
    if aug_x_train is not None and aug_y_train is not None:
        combined_x_train = np.concatenate([x_train, aug_x_train], axis=0)
        combined_y_train = np.concatenate([y_train, aug_y_train], axis=0)
        
        # Shuffle the combined training data
        combined_x_train, combined_y_train = shuffle(combined_x_train, combined_y_train, random_state=RANDOM_SEED)
    else:
        combined_x_train = x_train
        combined_y_train = y_train
    
    # Normalize the data
    combined_x_train, x_val, x_test = normalize_data(combined_x_train, x_val, x_test)
    
    # Encode the labels
    combined_y_train, y_val, y_test = encode_labels(combined_y_train, y_val, y_test)
    
    # Verify normalization
    print(f"Range of combined_x_train, x_val, x_test values: {np.min(combined_x_train)}-{np.max(combined_x_train)}, {np.min(x_val)}-{np.max(x_val)}, {np.min(x_test)}-{np.max(x_test)}")
    
    # Verify shapes and one-hot encoding of labels (samples, num_classes)
    print(f"Final shapes:")
    print(f"Training data   | X: {combined_x_train.shape}, Y: {combined_y_train.shape}")
    print(f"Validation data | X: {x_val.shape},   Y: {y_val.shape}")
    print(f"Test data       | X: {x_test.shape},  Y: {y_test.shape}")
    
    return combined_x_train, combined_y_train, x_val, y_val, x_test, y_test

def plot_training_history(history):
    """
    Function to plot training history
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
