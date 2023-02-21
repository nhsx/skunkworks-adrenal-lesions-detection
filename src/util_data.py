from typing import Optional, Union
import pandas as pd
from tqdm import tqdm
from glob import glob
import os

def listdir_nohidden(path):
    """
    function to list files full paths contained in the input path.
    """
    return glob(os.path.join(path, '*'))

def record_crop_array(data_filename: str, crop_current_data: pd.DataFrame) -> : pd.DataFrame:
    """
    Record the crop array in a DataFrame.
    """
    df_crop_current = pd.DataFrame(data=crop_current_data, index=[0])
    df_crop = pd.read_csv(data_filename,index_col=0)
    df_current_related = df_crop[(df_crop["filename_nii"]==df_crop_current["filename_nii"][0]) & (df_crop["adrenal_LR"]==df_crop_current["adrenal_LR"][0])]
    if df_current_related.empty:
        df_crop = df_crop.append(crop_current_data,ignore_index=True)
        print("New input: row added.")
    else:
        df_crop = df_crop.drop(df_crop[(df_crop["filename_nii"]==df_crop_current["filename_nii"][0]) & (df_crop["adrenal_LR"]==df_crop_current["adrenal_LR"][0])].index.values)
        df_crop = df_crop.append(crop_current_data,ignore_index=True)
        print("Existing input: row updated.")
    df_crop = df_crop.sort_values(['filename_nii', 'adrenal_LR'], ascending=[True, True],ignore_index=True)
    df_crop.to_csv(data_filename)
    del df_crop
    df_crop = pd.read_csv(data_filename,index_col=0)
    display(df_crop_current)
    display(df_crop[df_crop["filename_nii"]==df_crop_current["filename_nii"][0]])
    print(data_filename, "contains",df_crop.shape[0],"crop-array.")
    return df_crop

def check_corrupted(df: pd.DataFrame, filepath_col_name: str) -> list:
    """
    Double check if there are any corrupted file path
    """
    count_file =0
    count_no_such_file = 0
    drop_i_row = []
    
    for i in tqdm(range(len(df))):
        image_name = df[filepath_col_name].iloc[i]
        if os.path.exists(image_name): 
            count_file = count_file +1
        else:
            drop_i_row.append(i)
            count_no_such_file = count_no_such_file + 1
    
    print("Corrupted file path:",count_no_such_file)
    return drop_i_row
    
def find_binary_class_weights(series: Union[pd.Series, list]) -> list:
    """
    Find class weights.
    Scaling by total/2 helps keep the loss to a similar magnitude.
    The sum of the weights of all examples stays the same.
    """
    neg, pos = series.value_counts().sort_index()
    total_number_img = neg + pos
    weight_for_0 = (1 / neg) * (total_number_img / 2.0)
    weight_for_1 = (1 / pos) * (total_number_img / 2.0)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

def define_classes(series: Union[pd.Series, list]) -> [list, int]:
    """
    Define the classes and get number of classes.
    """
    classes_to_predict = sorted(series.unique())
    num_classes = len(classes_to_predict)
    return classes_to_predict,num_classes