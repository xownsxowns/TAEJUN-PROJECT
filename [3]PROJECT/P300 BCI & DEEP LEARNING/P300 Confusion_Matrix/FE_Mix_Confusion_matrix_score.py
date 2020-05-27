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

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
############ DOORLOCK ###############################
total_sum_raw_cnn_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_SVM_CNN_full_ch_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_raw_cnn_doorlock = total_sum_raw_cnn_doorlock + result

############ LAMP ###############################
total_sum_raw_cnn_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_SVM_CNN_full_ch_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_raw_cnn_lamp = total_sum_raw_cnn_lamp + result

############ SPEAKER ###############################
total_sum_raw_cnn_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_SVM_CNN_full_ch_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_raw_cnn_bs = total_sum_raw_cnn_bs + result


############ TOTAL ###############################
total_sum_raw_cnn = total_sum_raw_cnn_doorlock + total_sum_raw_cnn_lamp + total_sum_raw_cnn_bs

################ calculating #####################
## door lock
RAW_CNN_DL_precision = calculate_precision(total_sum_raw_cnn_doorlock)
RAW_CNN_DL_recall = calculate_recall(total_sum_raw_cnn_doorlock)
RAW_CNN_DL_F1 = calculate_f1_score(total_sum_raw_cnn_doorlock)

#### Electric light #####
RAW_CNN_EL_precision = calculate_precision(total_sum_raw_cnn_lamp)
RAW_CNN_EL_recall = calculate_recall(total_sum_raw_cnn_lamp)
RAW_CNN_EL_F1 = calculate_f1_score(total_sum_raw_cnn_lamp)

##### Bluetooth speaker
RAW_CNN_BS_precision = calculate_precision(total_sum_raw_cnn_bs)
RAW_CNN_BS_recall = calculate_recall(total_sum_raw_cnn_bs)
RAW_CNN_BS_F1 = calculate_f1_score(total_sum_raw_cnn_bs)

##### Total #########
RAW_CNN_precision = calculate_precision(total_sum_raw_cnn)
RAW_CNN_recall = calculate_recall(total_sum_raw_cnn)
RAW_CNN_F1 = calculate_f1_score(total_sum_raw_cnn)