import numpy as np
import pandas as pd

def calculate_precision(confusion_matrix):
    FP = confusion_matrix['1'][0]
    TP = confusion_matrix['1'][1]
    precision = TP / (TP+FP)
    return precision

def calculate_recall(confusion_matrix):
    FN = confusion_matrix['0'][1]
    TP = confusion_matrix['1'][1]
    recall = TP / (TP+FN)
    return recall

def calculate_f1_score(confusion_matrix):
    FP = confusion_matrix['1'][0]
    FN = confusion_matrix['0'][1]
    TP = confusion_matrix['1'][1]
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = 2 * ((precision*recall) / (precision+recall))
    return f1_score

############ DOORLOCK ###############################
# ROS
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/ROS/'
total_sum_ros_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ros_doorlock = total_sum_ros_doorlock + result
# SMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/SMOTE/'
total_sum_smote_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_smote_doorlock = total_sum_smote_doorlock + result
# B-SMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/'
total_sum_bsmote_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_bsmote_doorlock = total_sum_bsmote_doorlock + result
# SVMSMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/SVMSMOTE/'
total_sum_svmsmote_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svmsmote_doorlock = total_sum_svmsmote_doorlock + result
# ADASYN
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/ADASYN/'
total_sum_adasyn_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_adasyn_doorlock = total_sum_adasyn_doorlock + result
# RUS
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/RUS/'
total_sum_rus_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_rus_doorlock = total_sum_rus_doorlock + result
# NCR
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/'
total_sum_ncr_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ncr_doorlock = total_sum_ncr_doorlock + result
# Tomek
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/Tomek/'
total_sum_tomek_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_tomek_doorlock = total_sum_tomek_doorlock + result

######## EL ##############
# ROS
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/ROS/'
total_sum_ros_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ros_light = total_sum_ros_light + result
# SMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/SMOTE/'
total_sum_smote_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_smote_light = total_sum_smote_light + result
# B-SMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/'
total_sum_bsmote_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_bsmote_light = total_sum_bsmote_light + result
# SVMSMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/SVMSMOTE/'
total_sum_svmsmote_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svmsmote_light = total_sum_svmsmote_light + result
# ADASYN
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/ADASYN/'
total_sum_adasyn_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_adasyn_light = total_sum_adasyn_light + result
# RUS
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/RUS/'
total_sum_rus_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_rus_light = total_sum_rus_light + result
# NCR
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/'
total_sum_ncr_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ncr_light = total_sum_ncr_light + result
# Tomek
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/Tomek/'
total_sum_tomek_light = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_tomek_light = total_sum_tomek_light + result

################## BS #############
# ROS
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/ROS/'
total_sum_ros_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ros_bs = total_sum_ros_bs + result
# SMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/SMOTE/'
total_sum_smote_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_smote_bs = total_sum_smote_bs + result
# B-SMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/'
total_sum_bsmote_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_bsmote_bs = total_sum_bsmote_bs + result
# SVMSMOTE
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/SVMSMOTE/'
total_sum_svmsmote_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svmsmote_bs = total_sum_svmsmote_bs + result
# ADASYN
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/ADASYN/'
total_sum_adasyn_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_adasyn_bs = total_sum_adasyn_bs + result
# RUS
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/RUS/'
total_sum_rus_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_rus_bs = total_sum_rus_bs + result
# NCR
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/'
total_sum_ncr_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ncr_bs = total_sum_ncr_bs + result
# Tomek
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/Tomek/'
total_sum_tomek_bs = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_tomek_bs = total_sum_tomek_bs + result

total_sum_ros = total_sum_ros_doorlock + total_sum_ros_light + total_sum_ros_bs
total_sum_smote = total_sum_smote_doorlock + total_sum_smote_light + total_sum_smote_bs
total_sum_bsmote = total_sum_bsmote_doorlock + total_sum_bsmote_light + total_sum_bsmote_bs
total_sum_svmsmote = total_sum_svmsmote_doorlock + total_sum_svmsmote_light + total_sum_svmsmote_bs
total_sum_adasyn = total_sum_adasyn_doorlock + total_sum_adasyn_light + total_sum_adasyn_bs
total_sum_rus = total_sum_rus_doorlock + total_sum_rus_light + total_sum_rus_bs
total_sum_ncr = total_sum_ncr_doorlock + total_sum_ncr_light + total_sum_ncr_bs
total_sum_tomek = total_sum_tomek_doorlock + total_sum_tomek_light + total_sum_tomek_bs

## DL
SVM_DL_ros_precision = calculate_precision(total_sum_ros_doorlock)
SVM_DL_ros_recall = calculate_recall(total_sum_ros_doorlock)
SVM_DL_ros_F1 = calculate_f1_score(total_sum_ros_doorlock)

SVM_DL_smote_precision = calculate_precision(total_sum_smote_doorlock)
SVM_DL_smote_recall = calculate_recall(total_sum_smote_doorlock)
SVM_DL_smote_F1 = calculate_f1_score(total_sum_smote_doorlock)

SVM_DL_bsmote_precision = calculate_precision(total_sum_bsmote_doorlock)
SVM_DL_bsmote_recall = calculate_recall(total_sum_bsmote_doorlock)
SVM_DL_bsmote_F1 = calculate_f1_score(total_sum_bsmote_doorlock)

SVM_DL_svmsmote_precision = calculate_precision(total_sum_svmsmote_doorlock)
SVM_DL_svmsmote_recall = calculate_recall(total_sum_svmsmote_doorlock)
SVM_DL_svmsmote_F1 = calculate_f1_score(total_sum_svmsmote_doorlock)

SVM_DL_adasyn_precision = calculate_precision(total_sum_adasyn_doorlock)
SVM_DL_adasyn_recall = calculate_recall(total_sum_adasyn_doorlock)
SVM_DL_adasyn_F1 = calculate_f1_score(total_sum_adasyn_doorlock)

SVM_DL_rus_precision = calculate_precision(total_sum_rus_doorlock)
SVM_DL_rus_recall = calculate_recall(total_sum_rus_doorlock)
SVM_DL_rus_F1 = calculate_f1_score(total_sum_rus_doorlock)

SVM_DL_ncr_precision = calculate_precision(total_sum_ncr_doorlock)
SVM_DL_ncr_recall = calculate_recall(total_sum_ncr_doorlock)
SVM_DL_ncr_F1 = calculate_f1_score(total_sum_ncr_doorlock)

SVM_DL_tomek_precision = calculate_precision(total_sum_tomek_doorlock)
SVM_DL_tomek_recall = calculate_recall(total_sum_tomek_doorlock)
SVM_DL_tomek_F1 = calculate_f1_score(total_sum_tomek_doorlock)

## EL
SVM_EL_ros_precision = calculate_precision(total_sum_ros_light)
SVM_EL_ros_recall = calculate_recall(total_sum_ros_light)
SVM_EL_ros_F1 = calculate_f1_score(total_sum_ros_light)

SVM_EL_smote_precision = calculate_precision(total_sum_smote_light)
SVM_EL_smote_recall = calculate_recall(total_sum_smote_light)
SVM_EL_smote_F1 = calculate_f1_score(total_sum_smote_light)

SVM_EL_bsmote_precision = calculate_precision(total_sum_bsmote_light)
SVM_EL_bsmote_recall = calculate_recall(total_sum_bsmote_light)
SVM_EL_bsmote_F1 = calculate_f1_score(total_sum_bsmote_light)

SVM_EL_svmsmote_precision = calculate_precision(total_sum_svmsmote_light)
SVM_EL_svmsmote_recall = calculate_recall(total_sum_svmsmote_light)
SVM_EL_svmsmote_F1 = calculate_f1_score(total_sum_svmsmote_light)

SVM_EL_adasyn_precision = calculate_precision(total_sum_adasyn_light)
SVM_EL_adasyn_recall = calculate_recall(total_sum_adasyn_light)
SVM_EL_adasyn_F1 = calculate_f1_score(total_sum_adasyn_light)

SVM_EL_rus_precision = calculate_precision(total_sum_rus_light)
SVM_EL_rus_recall = calculate_recall(total_sum_rus_light)
SVM_EL_rus_F1 = calculate_f1_score(total_sum_rus_light)

SVM_EL_ncr_precision = calculate_precision(total_sum_ncr_light)
SVM_EL_ncr_recall = calculate_recall(total_sum_ncr_light)
SVM_EL_ncr_F1 = calculate_f1_score(total_sum_ncr_light)

SVM_EL_tomek_precision = calculate_precision(total_sum_tomek_light)
SVM_EL_tomek_recall = calculate_recall(total_sum_tomek_light)
SVM_EL_tomek_F1 = calculate_f1_score(total_sum_tomek_light)

## BS
SVM_BS_ros_precision = calculate_precision(total_sum_ros_bs)
SVM_BS_ros_recall = calculate_recall(total_sum_ros_bs)
SVM_BS_ros_F1 = calculate_f1_score(total_sum_ros_bs)

SVM_BS_smote_precision = calculate_precision(total_sum_smote_bs)
SVM_BS_smote_recall = calculate_recall(total_sum_smote_bs)
SVM_BS_smote_F1 = calculate_f1_score(total_sum_smote_bs)

SVM_BS_bsmote_precision = calculate_precision(total_sum_bsmote_bs)
SVM_BS_bsmote_recall = calculate_recall(total_sum_bsmote_bs)
SVM_BS_bsmote_F1 = calculate_f1_score(total_sum_bsmote_bs)

SVM_BS_svmsmote_precision = calculate_precision(total_sum_svmsmote_bs)
SVM_BS_svmsmote_recall = calculate_recall(total_sum_svmsmote_bs)
SVM_BS_svmsmote_F1 = calculate_f1_score(total_sum_svmsmote_bs)

SVM_BS_adasyn_precision = calculate_precision(total_sum_adasyn_bs)
SVM_BS_adasyn_recall = calculate_recall(total_sum_adasyn_bs)
SVM_BS_adasyn_F1 = calculate_f1_score(total_sum_adasyn_bs)

SVM_BS_rus_precision = calculate_precision(total_sum_rus_bs)
SVM_BS_rus_recall = calculate_recall(total_sum_rus_bs)
SVM_BS_rus_F1 = calculate_f1_score(total_sum_rus_bs)

SVM_BS_ncr_precision = calculate_precision(total_sum_ncr_bs)
SVM_BS_ncr_recall = calculate_recall(total_sum_ncr_bs)
SVM_BS_ncr_F1 = calculate_f1_score(total_sum_ncr_bs)

SVM_BS_tomek_precision = calculate_precision(total_sum_tomek_bs)
SVM_BS_tomek_recall = calculate_recall(total_sum_tomek_bs)
SVM_BS_tomek_F1 = calculate_f1_score(total_sum_tomek_bs)

## Total
SVM_ros_precision = calculate_precision(total_sum_ros)
SVM_ros_recall = calculate_recall(total_sum_ros)
SVM_ros_F1 = calculate_f1_score(total_sum_ros)

SVM_smote_precision = calculate_precision(total_sum_smote)
SVM_smote_recall = calculate_recall(total_sum_smote)
SVM_smote_F1 = calculate_f1_score(total_sum_smote)

SVM_bsmote_precision = calculate_precision(total_sum_bsmote)
SVM_bsmote_recall = calculate_recall(total_sum_bsmote)
SVM_bsmote_F1 = calculate_f1_score(total_sum_bsmote)

SVM_svmsmote_precision = calculate_precision(total_sum_svmsmote)
SVM_svmsmote_recall = calculate_recall(total_sum_svmsmote)
SVM_svmsmote_F1 = calculate_f1_score(total_sum_svmsmote)

SVM_adasyn_precision = calculate_precision(total_sum_adasyn)
SVM_adasyn_recall = calculate_recall(total_sum_adasyn)
SVM_adasyn_F1 = calculate_f1_score(total_sum_adasyn)

SVM_rus_precision = calculate_precision(total_sum_rus)
SVM_rus_recall = calculate_recall(total_sum_rus)
SVM_rus_F1 = calculate_f1_score(total_sum_rus)

SVM_ncr_precision = calculate_precision(total_sum_ncr)
SVM_ncr_recall = calculate_recall(total_sum_ncr)
SVM_ncr_F1 = calculate_f1_score(total_sum_ncr)

SVM_tomek_precision = calculate_precision(total_sum_tomek)
SVM_tomek_recall = calculate_recall(total_sum_tomek)
SVM_tomek_F1 = calculate_f1_score(total_sum_tomek)