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

######################## Train ###########################
#### ratio0
total_sum_ratio0 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio0' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio0 = total_sum_ratio0 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio0' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio0 = total_sum_ratio0 + result

#### ratio1
total_sum_ratio1 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio1' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio1 = total_sum_ratio1 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio1' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio1 = total_sum_ratio1 + result

#### ratio2
total_sum_ratio2 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio2' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio2 = total_sum_ratio2 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio2' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio2 = total_sum_ratio2 + result

#### ratio3
total_sum_ratio3 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio3' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio3 = total_sum_ratio3 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio3' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio3 = total_sum_ratio3 + result

#### ratio4
total_sum_ratio4 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio4' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio4 = total_sum_ratio4 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio4' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio4 = total_sum_ratio4 + result

#### ratio5
total_sum_ratio5 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio5' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio5 = total_sum_ratio5 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio5' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio5 = total_sum_ratio5 + result

#### ratio6
total_sum_ratio6 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio6' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio6 = total_sum_ratio6 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio6' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio6 = total_sum_ratio6 + result

#### ratio7
total_sum_ratio7 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio7' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio7 = total_sum_ratio7 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio7' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio7 = total_sum_ratio7 + result

#### ratio8
total_sum_ratio8 = np.zeros((2,2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio8' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio8 = total_sum_ratio8 + result

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio8' + '_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio8 = total_sum_ratio8 + result

######################## Test ###########################
#### ratio0
total_sum_ratio0_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio0' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio0_t = total_sum_ratio0_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio0' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio0_t = total_sum_ratio0_t + result

#### ratio1
total_sum_ratio1_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio1' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio1_t = total_sum_ratio1_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio1' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio1_t = total_sum_ratio1_t + result

#### ratio2
total_sum_ratio2_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio2' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio2_t = total_sum_ratio2_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio2' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio2_t = total_sum_ratio2_t + result

#### ratio3
total_sum_ratio3_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio3' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio3_t = total_sum_ratio3_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio3' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio3_t = total_sum_ratio3_t + result

#### ratio4
total_sum_ratio4_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio4' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio4_t = total_sum_ratio4_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio4' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio4_t = total_sum_ratio4_t + result

#### ratio5
total_sum_ratio5_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio5' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio5_t = total_sum_ratio5_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio5' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio5_t = total_sum_ratio5_t + result

#### ratio6
total_sum_ratio6_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio6' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio6_t = total_sum_ratio6_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio6' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio6_t = total_sum_ratio6_t + result

#### ratio7
total_sum_ratio7_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio7' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio7_t = total_sum_ratio7_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio7' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio7_t = total_sum_ratio7_t + result

#### ratio8
total_sum_ratio8_t = np.zeros((2, 2))
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31, 61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio8' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio8_t = total_sum_ratio8_t + result

for isub in range(1, 15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio8' + '_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_ratio8_t = total_sum_ratio8_t + result