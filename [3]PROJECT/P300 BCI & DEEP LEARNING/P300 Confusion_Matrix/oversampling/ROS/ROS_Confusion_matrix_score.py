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

path = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/result/CONFUSION/ROS/'

############ DOORLOCK ###############################
total_sum_cnn_ros_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'CNN_fullch_ros_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_ros_doorlock = total_sum_cnn_ros_doorlock + result

total_sum_svm_ros_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_ros_doorlock = total_sum_svm_ros_doorlock + result

############ LAMP ###############################
total_sum_cnn_ros_lamp = np.zeros((2,2))
for isub in range(46,61):
    file = path + 'CNN_fullch_ros_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_ros_lamp = total_sum_cnn_ros_lamp + result

total_sum_svm_ros_lamp = np.zeros((2,2))
for isub in range(46,61):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_ros_lamp = total_sum_svm_ros_lamp + result

############ SPEAKER ###############################
total_sum_cnn_ros_speaker = np.zeros((2,2))
for isub in range(1,15):
    file = path + 'CNN_BS_fullch_ros_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_ros_speaker = total_sum_cnn_ros_speaker + result

total_sum_svm_ros_speaker = np.zeros((2,2))
for isub in range(1,15):
    file = path + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_ros_speaker = total_sum_svm_ros_speaker + result

############ TOTAL ###############################
total_sum_cnn_ros = np.zeros((2,2))
for isub in range(31,61):
    file = path + 'CNN_fullch_ros_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_ros = total_sum_cnn_ros + result

for isub in range(1,15):
    file = path + 'CNN_BS_fullch_ros_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_ros = total_sum_cnn_ros + result

total_sum_svm_ros = np.zeros((2,2))
for isub in range(31,61):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_ros = total_sum_svm_ros + result

for isub in range(1,15):
    file = path + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_ros = total_sum_svm_ros + result

##
CNN_ROS_full_DL_precision = calculate_precision(total_sum_cnn_ros_doorlock)
CNN_ROS_full_DL_recall = calculate_recall(total_sum_cnn_ros_doorlock)
CNN_ROS_full_DL_F1 = calculate_f1_score(total_sum_cnn_ros_doorlock)
SVM_ROS_DL_precision = calculate_precision(total_sum_svm_ros_doorlock)
SVM_ROS_DL_recall = calculate_recall(total_sum_svm_ros_doorlock)
SVM_ROS_DL_F1 = calculate_f1_score(total_sum_svm_ros_doorlock)

CNN_ROS_full_EL_precision = calculate_precision(total_sum_cnn_ros_lamp)
CNN_ROS_full_EL_recall = calculate_recall(total_sum_cnn_ros_lamp)
CNN_ROS_full_EL_F1 = calculate_f1_score(total_sum_cnn_ros_lamp)
SVM_ROS_EL_precision = calculate_precision(total_sum_svm_ros_lamp)
SVM_ROS_EL_recall = calculate_recall(total_sum_svm_ros_lamp)
SVM_ROS_EL_F1 = calculate_f1_score(total_sum_svm_ros_lamp)

CNN_ROS_BS_full_precision = calculate_precision(total_sum_cnn_ros_speaker)
CNN_ROS_BS_full_recall = calculate_recall(total_sum_cnn_ros_speaker)
CNN_ROS_BS_full_F1 = calculate_f1_score(total_sum_cnn_ros_speaker)
SVM_ROS_BS_precision = calculate_precision(total_sum_svm_ros_speaker)
SVM_ROS_BS_recall = calculate_recall(total_sum_svm_ros_speaker)
SVM_ROS_BS_F1 = calculate_f1_score(total_sum_svm_ros_speaker)

CNN_ROS_full_precision = calculate_precision(total_sum_cnn_ros)
CNN_ROS_full_recall = calculate_recall(total_sum_cnn_ros)
CNN_ROS_full_F1 = calculate_f1_score(total_sum_cnn_ros)
SVM_ROS_precision = calculate_precision(total_sum_svm_ros)
SVM_ROS_recall = calculate_recall(total_sum_svm_ros)
SVM_ROS_F1 = calculate_f1_score(total_sum_svm_ros)

Total = [CNN_ROS_full_recall, CNN_ROS_full_precision, CNN_ROS_full_F1, SVM_ROS_recall, SVM_ROS_precision, SVM_ROS_F1,
         CNN_ROS_full_DL_recall, CNN_ROS_full_DL_precision, CNN_ROS_full_DL_F1, SVM_ROS_DL_recall, SVM_ROS_DL_precision, SVM_ROS_DL_F1,
         CNN_ROS_full_EL_recall, CNN_ROS_full_EL_precision, CNN_ROS_full_EL_F1, SVM_ROS_EL_recall, SVM_ROS_EL_precision, SVM_ROS_EL_F1,
         CNN_ROS_BS_full_recall, CNN_ROS_BS_full_precision, CNN_ROS_BS_full_F1, SVM_ROS_BS_recall, SVM_ROS_BS_precision, SVM_ROS_BS_F1]

df = pd.DataFrame(Total)
filename = 'ROS_Confusion matrix.csv'
df.to_csv(filename)