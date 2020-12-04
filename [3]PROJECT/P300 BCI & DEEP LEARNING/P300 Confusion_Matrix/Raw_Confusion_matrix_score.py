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

def calcualte_acc_score(confusion_matrix):
    FP = confusion_matrix['1'][0]
    FN = confusion_matrix['0'][1]
    TP = confusion_matrix['1'][1]
    TN = confusion_matrix['0'][0]
    acc = (TN+TP)/(FP+FN+TP+TN)
    return acc

path = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
#### RAW ####
test_recall = list()
test_precision = list()
test_acc = list()

total_sum = np.zeros((2,2))
for isub in range(31,60):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    test_recall.append(calculate_recall(result))
    test_precision.append(calculate_precision(result))
    test_acc.append(calcualte_acc_score(result))
    total_sum = total_sum + result
    print(result['1'][0]+result['1'][1]+result['0'][0]+result['0'][1])

#### RAW ####
for isub in range(1, 15):
    file = path + 'P300_Result_BS_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    test_recall.append(calculate_recall(result))
    test_precision.append(calculate_precision(result))
    test_acc.append(calcualte_acc_score(result))
    total_sum = total_sum + result
    print(result['1'][0]+result['1'][1]+result['0'][0]+result['0'][1])

test_recall_bsmote = list()
test_precision_bsmote = list()
test_acc_bsmote = list()

total_sum_bsmote = np.zeros((2,2))
for isub in range(31,60):
    file = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling' \
               '/CONFUSION/B-SMOTE/P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    test_recall_bsmote.append(calculate_recall(result))
    test_precision_bsmote.append(calculate_precision(result))
    test_acc_bsmote.append(calcualte_acc_score(result))
    total_sum_bsmote = total_sum_bsmote + result
    print(result['1'][0]+result['1'][1]+result['0'][0]+result['0'][1])

#### RAW ####
for isub in range(1, 15):
    file = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE' \
               '/P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    test_recall_bsmote.append(calculate_recall(result))
    test_precision_bsmote.append(calculate_precision(result))
    test_acc_bsmote.append(calcualte_acc_score(result))
    total_sum_bsmote = total_sum_bsmote + result
    print(result['1'][0]+result['1'][1]+result['0'][0]+result['0'][1])

