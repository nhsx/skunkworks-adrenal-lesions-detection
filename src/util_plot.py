from typing import Optional, Union
import matplotlib.pylab as plt
import numpy as np
import cv2
import os
import random
import itertools
import pandas as pd
from scipy import ndimage
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import expand_dims
from tensorflow.keras import Sequential

# set global values for matplotlib
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
# set plots font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)

def show_one_image(image_path: str):
    """
    Show the image of the given path.
    """
    image = cv2.imread(image_path,0)
    image_R = ndimage.rotate(image, 90, reshape=True)
    # image = np.load(image_path)
    plt.figure(figsize=(8,8))
    plt.imshow(image_R,cmap=plt.cm.bone)
    plt.show()
    return

def show_augmentation_layers(image_path: str,data_augmentation_layers: Sequential) -> None:
    image = Image.open(image_path)
    image = ndimage.rotate(image, 90, reshape=True)
    # image = np.load(image_path)
    image = expand_dims(image, 0)
    plt.figure(figsize=(10, 8))
    for i in range(9):
        augmented_image = data_augmentation_layers(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0], cmap=plt.cm.bone)
        plt.axis("off")
    plt.show()
        
def plot_learning_curve(HISTORY_HISTORY_ACCURACY: list,
                        HISTORY_HISTORY_VALACCURACY: list,
                        HISTORY_HISTORY_LOSS: list,
                        HISTORY_HISTORY_VALLOSS: list,
                        epochs) -> None:
    """
    Plot the learning curves: accuracy and loss.
    """
    plt.figure(figsize=(10,10), linewidth=1.5)
    plt.subplot(2,1,1)
    plt.plot([-1,-1],[max(max(HISTORY_HISTORY_ACCURACY)),max(max(HISTORY_HISTORY_ACCURACY))],color="black",linestyle="-")
    plt.plot([-1,-1],[max(max(HISTORY_HISTORY_ACCURACY)),max(max(HISTORY_HISTORY_ACCURACY))],color="black",linestyle="-.")
    plt.legend(['Train','Validation'], loc='lower right')
    plt.plot(range(1,epochs+1),np.array(HISTORY_HISTORY_ACCURACY).T)
    plt.gca().set_prop_cycle(None)
    plt.plot(range(1,epochs+1),np.array(HISTORY_HISTORY_VALACCURACY).T,"-.")
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xlim([0.8, epochs+0.2])
    # summarize history for loss
    plt.subplot(2,1,2)
    plt.plot(range(1,epochs+1),np.array(HISTORY_HISTORY_LOSS).T)
    plt.gca().set_prop_cycle(None)
    plt.plot(range(1,epochs+1),np.array(HISTORY_HISTORY_VALLOSS).T,"-.")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim([0.8, epochs+0.2])
    plt.tight_layout()
    
def predict_image(filepath: str, model) -> int:
    """
    This function give prediction on one image
    """
    img = np.array([np.array(Image.open(filepath))])
    result = model.predict(img)
    return np.argmax(result)

def test_image(idx:int, model, X: Union[pd.Series, list], y: Union[pd.Series, list]):
    """
    This function do prediction test on one image
    """
    result = predict_image(X.iloc[idx], model)
    expected_output = y.iloc[idx]
    image_name = os.path.basename(X.iloc[idx])
    print("Image:{}".format(image_name))
    print("expected output: {}".format(expected_output))
    print("predicted output:{}".format(result))
    print("-----------------------------------")
    return

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues) -> None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)#, vmin=0,vmax=20)
    plt.title(title)
    plt.colorbar(shrink=0.72, aspect=18)
    # plt.clim(0,20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def plot_roc_curve(true_classes, test_predictions_ravel):
    fpr, tpr, thresholds = roc_curve(true_classes, test_predictions_ravel)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, 'b',linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='k')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(['ROC curve', 'Trivial model (AUC 0.5)'],loc='lower right')
    plt.show()
    AUC_ROC_SCORE = roc_auc_score(true_classes,test_predictions_ravel,average='weighted')
    print("AUC-ROC score = ",AUC_ROC_SCORE)
    return AUC_ROC_SCORE


def pick_up_right_prediction(idx: int, model, X: Union[pd.Series, list], y: Union[pd.Series, list]) -> bool:
    """
    This function pick up any right prediction through the image index (idx)
    """
    result = predict_image(X.iloc[idx],model)
    expected_output = y.iloc[idx]
    image_name = os.path.basename(X.iloc[idx])
    if result == expected_output:
        print("Image:{}".format(image_name))
        if expected_output == 0:
            print("expected output:  Others     = {}".format(expected_output))
        else:
            print("expected output:  Whole lung = {}".format(expected_output))
        if result == 0:
            print("predicted output: Others     = {}".format(result))
        else:
            print("predicted output: Whole lung = {}".format(result))
        image = cv2.imread(X.iloc[idx],0)
        image_R = ndimage.rotate(image, 180, reshape=False)
        plt.figure(figsize=(8,8))
        plt.imshow(image_R,cmap=plt.cm.bone)
        plt.show()
        print("---------------------------------------------")
    else:
        pass
    return (result == expected_output)
    
def pick_up_wrong_prediction(idx: int, model, X: Union[pd.Series, list], y: Union[pd.Series, list]) -> bool:
    """
    This function pick up any wrong prediction through the image index (idx)
    """
    result = predict_image(X.iloc[idx],model)
    expected_output = y.iloc[idx]
    image_name = os.path.basename(X.iloc[idx])
    if result == expected_output:
        pass
    else:
        print("Image:{}".format(image_name))
        if expected_output == 0:
            print("expected output:  Others     = {}".format(expected_output))
        else:
            print("expected output:  Whole lung = {}".format(expected_output))
        if result == 0:
            print("predicted output: Others     = {}".format(result))
        else:
            print("predicted output: Whole lung = {}".format(result))
        image = cv2.imread(X.iloc[idx],0)
        image_R = ndimage.rotate(image, 180, reshape=False)
        plt.figure(figsize=(8,8))
        plt.imshow(image_R,cmap=plt.cm.bone)
        plt.show()
        print("---------------------------------------------")
    return (result == expected_output)

def display_right_prediction(display_number: int, model, X: Union[pd.Series, list], y: Union[pd.Series, list]):
    """
    Display correctly predicted images
    """
    right_count = 0
    for idx in random.sample(range(0, len(y)), len(y)):
        if right_count < display_number:
            outcome = pick_up_right_prediction(idx, model, X, y)
            if outcome == True:
                right_count += 1
    return
    
def display_wrong_prediction(display_number: int, model, X: Union[pd.Series, list], y: Union[pd.Series, list]):
    """
    Display wrongly predicted images
    """
    right_count = 0
    for idx in random.sample(range(0, len(y)), len(y)):
        if right_count < display_number:
            outcome = pick_up_wrong_prediction(idx, model, X, y)
            if outcome == False:
                right_count += 1
    return
    
def plot_op(op_list,accuracy_list):
    plt.plot(op_list,accuracy_list)
    plt.xlabel('Operating point')
    plt.ylabel('Accuracy score (on CT scan)')
    plt.show()
    
    