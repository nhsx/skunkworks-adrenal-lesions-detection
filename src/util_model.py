from typing import Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg
from tensorflow.io import decode_raw
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB1, EfficientNetB2 , EfficientNetB5


def train_val_test_split(X: Union[pd.Series, list], 
                         y: Union[pd.Series, list],
                         size_ratio: list,
                         random_state: int) -> pd.Series:
    """
    Split to the training, validation, and test set.
    """
    #check size_ratio = 1.0
    if sum(size_ratio) == 1.0:
        val_test_size = 1.0 - size_ratio[0]
        val_test_ratio = size_ratio[2]/(size_ratio[1]+size_ratio[2])
        X_train, X_sub, y_train, y_sub = train_test_split(X, y, test_size=val_test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_sub, y_sub, test_size=val_test_ratio, random_state=random_state)
        print('Number of images in the train set: ',len(X_train))
        print('Number of images in the validation set: ',len(X_val))
        print('Number of images in the test set: ',len(X_test))
    else:
        print("!! size_ratio must be equals to 1.0.")
        print("Split fail.")
    return X_train, y_train, X_val, y_val, X_test, y_test
    
def load_image_and_label_from_path(image_path: str, label: int):
    """
    Function to load the image and label from the path
    """
    img = read_file(image_path)
    img = decode_jpeg(img, channels=3)
    return img, label

def get_data_augmentation_layers_keras(aug_rotation: tuple, aug_zoom: tuple, aug_contrast:tuple):
    """
    Get the image augmentations augmentation layer (keras)
    """
    data_augmentation_layers = Sequential([
    #                     layers.experimental.preprocessing.RandomCrop(height=image_size, width=image_size),
    #                     layers.experimental.preprocessing.RandomFlip("horizontal"),
                        layers.experimental.preprocessing.RandomRotation(aug_rotation, fill_mode="constant"),
                        layers.experimental.preprocessing.RandomZoom(aug_zoom, fill_mode="constant"),
                        layers.experimental.preprocessing.RandomContrast(aug_contrast)                
                        ])
    return data_augmentation_layers
    
def get_imgaug_augmentation(aug_coarsedropout: tuple, aug_invert: tuple) -> iaa.meta.Sequential:
    """
    Get the image augmentations augmentation (imgaug)
    """
    seq = iaa.Sequential([
                    iaa.CoarseDropout(aug_coarsedropout),
                    iaa.Invert(aug_invert)
                    ])
    return seq

def create_model_enb1(input_shape,num_classes,dropout_rate,data_augmentation_layers):
    """ 
    Build the EfficientNetB1 model
    """
    enb1 = EfficientNetB1(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape, 
                          drop_connect_rate=dropout_rate
                          )

    inputs = Input(shape=input_shape)
    augmented = data_augmentation_layers(inputs)
    enb1 = enb1(augmented)
    pooling = GlobalAveragePooling2D()(enb1)
    dropout = Dropout(dropout_rate)(pooling)
    outputs = Dense(num_classes, activation="softmax")(dropout)
    model_enb1 = Model(inputs=inputs, outputs=outputs)
    return model_enb1

def create_model_enb2(input_shape,num_classes,dropout_rate,data_augmentation_layers):
    """ 
    Build the EfficientNetB2 model
    """
    enb2 = EfficientNetB2(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape, 
                          drop_connect_rate=dropout_rate
                          )

    inputs = Input(shape=input_shape)
    augmented = data_augmentation_layers(inputs)
    enb2 = enb2(augmented)
    pooling = GlobalAveragePooling2D()(enb2)
    dropout = Dropout(dropout_rate)(pooling)
    outputs = Dense(num_classes, activation="softmax")(dropout)
    model_enb2 = Model(inputs=inputs, outputs=outputs)
    return model_enb2
    
def create_model_enb5(input_shape,num_classes,dropout_rate,data_augmentation_layers):
    """ 
    Build the EfficientNetB5 model
    """
    enb5 = EfficientNetB5(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape, 
                          drop_connect_rate=dropout_rate
                          )

    inputs = Input(shape=input_shape)
    augmented = data_augmentation_layers(inputs)
    enb5 = enb5(augmented)
    pooling = GlobalAveragePooling2D()(enb5)
    dropout = Dropout(dropout_rate)(pooling)
    outputs = Dense(num_classes, activation="softmax")(dropout)
    model_enb5 = Model(inputs=inputs, outputs=outputs)
    return model_enb5
    