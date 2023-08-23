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

path = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/result/CONFUSION/Tomek/'

############ DOORLOCK ###############################
total_sum_cnn_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'CNN_fullch_tomek_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_doorlock = total_sum_cnn_doorlock + result

total_sum_svm_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_doorlock = total_sum_svm_doorlock + result

############ LAMP ###############################
total_sum_cnn_lamp = np.zeros((2,2))
for isub in range(46,61):
    file = path + 'CNN_fullch_tomek_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_lamp = total_sum_cnn_lamp + result

total_sum_svm_lamp = np.zeros((2,2))
for isub in range(46,61):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_lamp = total_sum_svm_lamp + result

############ SPEAKER ###############################
total_sum_cnn_speaker = np.zeros((2,2))
for isub in range(1,15):
    file = path + 'CNN_BS_fullch_tomek_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_speaker = total_sum_cnn_speaker + result

total_sum_svm_speaker = np.zeros((2,2))
for isub in range(1,15):
    file = path + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm_speaker = total_sum_svm_speaker + result

############ TOTAL ###############################
total_sum_cnn = np.zeros((2,2))
for isub in range(31,61):
    file = path + 'CNN_fullch_tomek_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn = total_sum_cnn + result

for isub in range(1,15):
    file = path + 'CNN_BS_fullch_tomek_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn = total_sum_cnn + result

total_sum_svm = np.zeros((2,2))
for isub in range(31,61):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm = total_sum_svm + result

for isub in range(1,15):
    file = path + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm = total_sum_svm + result

##
CNN_full_DL_precision = calculate_precision(total_sum_cnn_doorlock)
CNN_full_DL_recall = calculate_recall(total_sum_cnn_doorlock)
CNN_full_DL_F1 = calculate_f1_score(total_sum_cnn_doorlock)
SVM_DL_precision = calculate_precision(total_sum_svm_doorlock)
SVM_DL_recall = calculate_recall(total_sum_svm_doorlock)
SVM_DL_F1 = calculate_f1_score(total_sum_svm_doorlock)

CNN_full_EL_precision = calculate_precision(total_sum_cnn_lamp)
CNN_full_EL_recall = calculate_recall(total_sum_cnn_lamp)
CNN_full_EL_F1 = calculate_f1_score(total_sum_cnn_lamp)
SVM_EL_precision = calculate_precision(total_sum_svm_lamp)
SVM_EL_recall = calculate_recall(total_sum_svm_lamp)
SVM_EL_F1 = calculate_f1_score(total_sum_svm_lamp)

CNN_BS_full_precision = calculate_precision(total_sum_cnn_speaker)
CNN_BS_full_recall = calculate_recall(total_sum_cnn_speaker)
CNN_BS_full_F1 = calculate_f1_score(total_sum_cnn_speaker)
SVM_BS_precision = calculate_precision(total_sum_svm_speaker)
SVM_BS_recall = calculate_recall(total_sum_svm_speaker)
SVM_BS_F1 = calculate_f1_score(total_sum_svm_speaker)

CNN_full_precision = calculate_precision(total_sum_cnn)
CNN_full_recall = calculate_recall(total_sum_cnn)
CNN_full_F1 = calculate_f1_score(total_sum_cnn)
SVM_precision = calculate_precision(total_sum_svm)
SVM_recall = calculate_recall(total_sum_svm)
SVM_F1 = calculate_f1_score(total_sum_svm)

Total = [CNN_full_recall, CNN_full_precision, CNN_full_F1, SVM_recall, SVM_precision, SVM_F1,
         CNN_full_DL_recall, CNN_full_DL_precision, CNN_full_DL_F1, SVM_DL_recall, SVM_DL_precision, SVM_DL_F1,
         CNN_full_EL_recall, CNN_full_EL_precision, CNN_full_EL_F1, SVM_EL_recall, SVM_EL_precision, SVM_EL_F1,
         CNN_BS_full_recall, CNN_BS_full_precision, CNN_BS_full_F1, SVM_BS_recall, SVM_BS_precision, SVM_BS_F1]

df = pd.DataFrame(Total)
filename = 'Tomek_Confusion matrix.csv'
df.to_csv(filename)