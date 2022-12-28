from typing import Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.util_plot import plot_confusion_matrix, plot_roc_curve, plot_op

def cal_sensitivity_specificity(cm,print_value=0):
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    sensitivity = TP / (TP+FN)
    specificity = TN / (TN+FP)
    precision = TP / (TP+FP)
    prevalence = (TP + FN)/(TP+TN+FP+FN)
    PPV = (sensitivity*prevalence)/((sensitivity*prevalence)+((1.0-specificity)*(1.0-prevalence)))
    NPV = (specificity*(1-prevalence))/((specificity*(1-prevalence))+((1-sensitivity)*prevalence))
    if print_value != 0:
        print("Sensitivity = ",sensitivity)
        print("Specificity = ",specificity)
        print("Precision   = ",precision)
        print("Prevalence  = ",prevalence)
        print("PPV         = ",PPV)
        print("NPV         = ",NPV)
    return sensitivity, specificity, precision, PPV, NPV
    
def analysis_1fold(true_classes,class_labels,test_prediction,predicted_classes):
    # Analysis on the classification performance on the test set
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report) 
    test_acc_score = accuracy_score(true_classes,predicted_classes)
    print("Accuracy    = ",test_acc_score)
    #plot confusion matrix
    cm = confusion_matrix(true_classes,predicted_classes)
    sens, spec, prec, PPV, NPV = cal_sensitivity_specificity(cm,print_value=1)
    plot_confusion_matrix(cm, class_labels, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues)
    #plot the ROC curve
    auc_roc_score = plot_roc_curve(true_classes, test_prediction)
    plt.show()
    return
    
def analysis_1fold_95CT(true_classes,class_labels,test_prediction,predicted_classes,nsample=5000):
    #plot the ROC curve
    auc_roc_score = plot_roc_curve(true_classes, test_prediction)
    # get 95% confidence interval
    CI95_all, auc_values_all = acu_95CI_bootstrap(true_classes, test_prediction, nsample)
    CI95_sen_all, CI95_spe_all = SensitivitySpecificity_95CI_bootstrap(true_classes, predicted_classes, nsample)
    # classification performance
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report) 
    balanced_accuracy = balanced_accuracy_score(true_classes,predicted_classes)
    print("Accuracy - Balanced = ",balanced_accuracy)
    #plot confusion matrix
    cm = confusion_matrix(true_classes,predicted_classes)
    sens, spec, prec, PPV, NPV = cal_sensitivity_specificity(cm,1)
    plot_confusion_matrix(cm, class_labels, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues)
    plt.show()
    return
    
def cal_roc_score(true_classes, test_predictions):
    AUC_ROC_SCORE = roc_auc_score(true_classes,test_predictions,average='weighted')
    #print("AUC-ROC score = ",AUC_ROC_SCORE)
    return AUC_ROC_SCORE

def print_roc_score(true_classes, test_predictions):
    AUC_ROC_SCORE = cal_roc_score(true_classes, test_predictions)
    print("------------------------------------------------------")
    print("Independent from cutoff value.")
    print("AUC-ROC score = ",AUC_ROC_SCORE)
    print("------------------------------------------------------")
    return AUC_ROC_SCORE

def cal_best_roc_score(true_classes, test_predictions, cretical_point):
    predicted_classes_max = np.where(test_predictions<cretical_point,0,1)
    AUC_ROC_SCORE_AT_CUTOFF = roc_auc_score(true_classes,predicted_classes_max,average='macro')
    #print("AUC-ROC score = ",AUC_ROC_SCORE)
    return AUC_ROC_SCORE_AT_CUTOFF

def show_classification_dist(dataframe, column_name, bins):
    """
    show the distribution.
    """
    plt.figure(figsize=(10,6))
    sns.displot(data = dataframe, x=column_name, bins = bins, kde=True,
               height=3.5,aspect=1.618)
    plt.xlabel(column_name,fontsize=13)
    plt.ylabel("Counts",fontsize=13)
    plt.show()
    return

def acu_95CI_bootstrap(true_classes, test_predictions, nsample=5000):
    auc_values = [] 
    for i in range(nsample):
        idx = np.random.randint(test_predictions.shape[0], size=test_predictions.shape[0])
        roc_auc = roc_auc_score(true_classes[idx], test_predictions[idx])
        auc_values.append(roc_auc)    
    CI95 = np.percentile(auc_values, (2.5, 97.5)) #The 95% CI
    print("95% CI = ",CI95)
    plt.figure(figsize=(10,6))
    sns.displot(data = auc_values, bins = 20, kde=True, height=3.5,aspect=1.618)
    plt.xlabel("ROC_AUC Score",fontsize=13)
    plt.ylabel("",fontsize=13)
    plt.show()
    return CI95, auc_values

def SensitivitySpecificity_95CI_bootstrap(true_classes, predicted_classes, nsample=5000):
    sen_values = []
    spe_values = []
    for i in range(nsample):
        idx = np.random.randint(predicted_classes.shape[0], size=predicted_classes.shape[0])
        cm = confusion_matrix(true_classes[idx],predicted_classes[idx])
        sens, spec, prec, PPV, NPV = cal_sensitivity_specificity(cm,print_value=0)
        sen_values.append(sens)
        spe_values.append(spec)
    CI95_sen = np.percentile(sen_values, (2.5, 97.5)) #The 95% CI
    CI95_spe = np.percentile(spe_values, (2.5, 97.5)) #The 95% CI
    print("Sensitivity 95% CI = ", CI95_sen)
    print("Specificity 95% CI = ", CI95_spe)
    return CI95_sen, CI95_spe
    
def get_df_image_performance(df,op):
    image_list = list(set(df["image_name"]))
    df_ctscan_col_list = ["PatientName","image_name","abnormal",
                            "img_test_prediction_avg_fold1","img_predicted_classes_avg_fold1","img_predicted_classes_fold1",
                            "img_test_prediction_avg_fold2","img_predicted_classes_avg_fold2","img_predicted_classes_fold2",
                            "img_test_prediction_avg_fold3","img_predicted_classes_avg_fold3","img_predicted_classes_fold3",
                            "img_test_prediction_avg_fold4","img_predicted_classes_avg_fold4","img_predicted_classes_fold4",
                            "img_test_prediction_avg_fold5","img_predicted_classes_avg_fold5","img_predicted_classes_fold5",
                            "img_test_prediction_avg_5fold","img_predicted_classes_avg_5fold","img_predicted_classes_5fold"]
    df_image_performance = pd.DataFrame(columns=df_ctscan_col_list)
    for image_name in image_list:
        df_image = df[df["image_name"] ==  image_name]
        df_current_image_performance = pd.DataFrame(columns=df_ctscan_col_list)
        df_current_image_performance["PatientName"] = [df_image.iloc[0].PatientName]
        df_current_image_performance["image_name"] = [df_image.iloc[0].image_name]
        df_current_image_performance["abnormal"] = [df_image.iloc[0].abnormal]
        df_current_image_performance["img_test_prediction_avg_fold1"] = np.average(df_image.test_prediction_model_fold1)
        df_current_image_performance["img_predicted_classes_avg_fold1"] = np.average(df_image.predicted_classes_model_fold1)
        df_current_image_performance["img_predicted_classes_fold1"] = np.where(df_current_image_performance.img_predicted_classes_avg_fold1<op,0,1)

        df_current_image_performance["img_test_prediction_avg_fold2"] = np.average(df_image.test_prediction_model_fold2)
        df_current_image_performance["img_predicted_classes_avg_fold2"] = np.average(df_image.predicted_classes_model_fold2)
        df_current_image_performance["img_predicted_classes_fold2"] = np.where(df_current_image_performance.img_predicted_classes_avg_fold2<op,0,1)

        df_current_image_performance["img_test_prediction_avg_fold3"] = np.average(df_image.test_prediction_model_fold3)
        df_current_image_performance["img_predicted_classes_avg_fold3"] = np.average(df_image.predicted_classes_model_fold3)
        df_current_image_performance["img_predicted_classes_fold3"] = np.where(df_current_image_performance.img_predicted_classes_avg_fold3<op,0,1)

        df_current_image_performance["img_test_prediction_avg_fold4"] = np.average(df_image.test_prediction_model_fold4)
        df_current_image_performance["img_predicted_classes_avg_fold4"] = np.average(df_image.predicted_classes_model_fold4)
        df_current_image_performance["img_predicted_classes_fold4"] = np.where(df_current_image_performance.img_predicted_classes_avg_fold4<op,0,1)

        df_current_image_performance["img_test_prediction_avg_fold5"] = np.average(df_image.test_prediction_model_fold5)
        df_current_image_performance["img_predicted_classes_avg_fold5"] = np.average(df_image.predicted_classes_model_fold5)
        df_current_image_performance["img_predicted_classes_fold5"] = np.where(df_current_image_performance.img_predicted_classes_avg_fold5<op,0,1)

        df_current_image_performance["img_test_prediction_avg_5fold"] = np.average(df_image.test_prediction_5fold)
        df_current_image_performance["img_predicted_classes_avg_5fold"] = np.average(df_image.predicted_classes_5fold)
        df_current_image_performance["img_predicted_classes_5fold"] = np.where(df_current_image_performance.img_predicted_classes_avg_5fold<op,0,1)

        df_image_performance = df_image_performance.append(df_current_image_performance, ignore_index=True)
    return df_image_performance
    
def cal_best_op(df,nprange):
    op_list = []
    accuracy_list = []
    for op in tqdm(nprange):
        df_image_performance_optest = get_df_image_performance(df,op)
        # Get most likely class
        test_prediction_avg_img = df_image_performance_optest.img_test_prediction_avg_5fold
        predicted_classes_avg_img = np.array(df_image_performance_optest.img_predicted_classes_avg_5fold)
        predicted_classes_img = np.array(df_image_performance_optest.img_predicted_classes_5fold, dtype=int)
        true_classes = np.array(df_image_performance_optest.abnormal, dtype=int)
        class_labels = ['Normal','Abnormal'] # label 0 = Normal; 1 = Abnormal
        accuracy = accuracy_score(true_classes,predicted_classes_img)
        op_list.append(op)
        accuracy_list.append(accuracy)
        
    # get best operating point
    plot_op(op_list,accuracy_list)
    op_for_best_acc = op_list[np.argmax(accuracy_list)]
    print("Max accuracy score at operating point =",op_for_best_acc)
    return op_for_best_acc







