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

def calculate_acc(confusion_matrix):
    FP = confusion_matrix['1'][0]
    FN = confusion_matrix['0'][1]
    TP = confusion_matrix['1'][1]
    TN = confusion_matrix['0'][0]
    acc = (TP+TN) / (TP+TN+FP+FN)
    return acc
############## Train ##################
# Raw
train_raw_recall = list()
train_raw_precision = list()
train_raw_f1 = list()
train_raw_acc = list()

path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_raw_recall.append(recall)
    train_raw_precision.append(precision)
    train_raw_f1.append(f1)
    train_raw_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_raw_recall.append(recall)
    train_raw_precision.append(precision)
    train_raw_f1.append(f1)
    train_raw_acc.append(acc)

# Ratio0
train_ratio0_recall = list()
train_ratio0_precision = list()
train_ratio0_f1 = list()
train_ratio0_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio0_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio0_recall.append(recall)
    train_ratio0_precision.append(precision)
    train_ratio0_f1.append(f1)
    train_ratio0_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio0_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio0_recall.append(recall)
    train_ratio0_precision.append(precision)
    train_ratio0_f1.append(f1)
    train_ratio0_acc.append(acc)

# Ratio1
train_ratio1_recall = list()
train_ratio1_precision = list()
train_ratio1_f1 = list()
train_ratio1_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio1_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio1_recall.append(recall)
    train_ratio1_precision.append(precision)
    train_ratio1_f1.append(f1)
    train_ratio1_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio1_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio1_recall.append(recall)
    train_ratio1_precision.append(precision)
    train_ratio1_f1.append(f1)
    train_ratio1_acc.append(acc)

# Ratio2
train_ratio2_recall = list()
train_ratio2_precision = list()
train_ratio2_f1 = list()
train_ratio2_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio2_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio2_recall.append(recall)
    train_ratio2_precision.append(precision)
    train_ratio2_f1.append(f1)
    train_ratio2_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio2_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio2_recall.append(recall)
    train_ratio2_precision.append(precision)
    train_ratio2_f1.append(f1)
    train_ratio2_acc.append(acc)

# Ratio3
train_ratio3_recall = list()
train_ratio3_precision = list()
train_ratio3_f1 = list()
train_ratio3_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio3_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio3_recall.append(recall)
    train_ratio3_precision.append(precision)
    train_ratio3_f1.append(f1)
    train_ratio3_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio3_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio3_recall.append(recall)
    train_ratio3_precision.append(precision)
    train_ratio3_f1.append(f1)
    train_ratio3_acc.append(acc)

# Ratio4
train_ratio4_recall = list()
train_ratio4_precision = list()
train_ratio4_f1 = list()
train_ratio4_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio4_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio4_recall.append(recall)
    train_ratio4_precision.append(precision)
    train_ratio4_f1.append(f1)
    train_ratio4_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio4_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio4_recall.append(recall)
    train_ratio4_precision.append(precision)
    train_ratio4_f1.append(f1)
    train_ratio4_acc.append(acc)

# Ratio5
train_ratio5_recall = list()
train_ratio5_precision = list()
train_ratio5_f1 = list()
train_ratio5_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio5_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio5_recall.append(recall)
    train_ratio5_precision.append(precision)
    train_ratio5_f1.append(f1)
    train_ratio5_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio5_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio5_recall.append(recall)
    train_ratio5_precision.append(precision)
    train_ratio5_f1.append(f1)
    train_ratio5_acc.append(acc)

# Ratio6
train_ratio6_recall = list()
train_ratio6_precision = list()
train_ratio6_f1 = list()
train_ratio6_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio6_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio6_recall.append(recall)
    train_ratio6_precision.append(precision)
    train_ratio6_f1.append(f1)
    train_ratio6_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio6_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio6_recall.append(recall)
    train_ratio6_precision.append(precision)
    train_ratio6_f1.append(f1)
    train_ratio6_acc.append(acc)

# Ratio7
train_ratio7_recall = list()
train_ratio7_precision = list()
train_ratio7_f1 = list()
train_ratio7_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio7_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio7_recall.append(recall)
    train_ratio7_precision.append(precision)
    train_ratio7_f1.append(f1)
    train_ratio7_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio7_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio7_recall.append(recall)
    train_ratio7_precision.append(precision)
    train_ratio7_f1.append(f1)
    train_ratio7_acc.append(acc)

# Ratio8
train_ratio8_recall = list()
train_ratio8_precision = list()
train_ratio8_f1 = list()
train_ratio8_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio8_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio8_recall.append(recall)
    train_ratio8_precision.append(precision)
    train_ratio8_f1.append(f1)
    train_ratio8_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio8_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_ratio8_recall.append(recall)
    train_ratio8_precision.append(precision)
    train_ratio8_f1.append(f1)
    train_ratio8_acc.append(acc)

# Full bsmote
train_bs_recall = list()
train_bs_precision = list()
train_bs_f1 = list()
train_bs_acc = list()
path2= 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_bs_recall.append(recall)
    train_bs_precision.append(precision)
    train_bs_f1.append(f1)
    train_bs_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_confusion_' + str(isub) + '_train.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    train_bs_recall.append(recall)
    train_bs_precision.append(precision)
    train_bs_f1.append(f1)
    train_bs_acc.append(acc)

train_recall = np.concatenate([np.reshape(train_raw_recall,(44,1)), np.reshape(train_ratio0_recall,(44,1)), np.reshape(train_ratio1_recall,(44,1)),
                               np.reshape(train_ratio2_recall,(44,1)),np.reshape(train_ratio3_recall,(44,1)), np.reshape(train_ratio4_recall,(44,1)),
                               np.reshape(train_ratio5_recall,(44,1)),np.reshape(train_ratio6_recall,(44,1)), np.reshape(train_ratio7_recall,(44,1)),
                               np.reshape(train_ratio8_recall,(44,1)),np.reshape(train_bs_recall,(44,1))],axis=1)
train_precision = np.concatenate([np.reshape(train_raw_precision,(44,1)), np.reshape(train_ratio0_precision,(44,1)), np.reshape(train_ratio1_precision,(44,1)),
                               np.reshape(train_ratio2_precision,(44,1)),np.reshape(train_ratio3_precision,(44,1)), np.reshape(train_ratio4_precision,(44,1)),
                               np.reshape(train_ratio5_precision,(44,1)),np.reshape(train_ratio6_precision,(44,1)), np.reshape(train_ratio7_precision,(44,1)),
                               np.reshape(train_ratio8_precision,(44,1)),np.reshape(train_bs_precision,(44,1))],axis=1)
train_f1 = np.concatenate([np.reshape(train_raw_f1,(44,1)), np.reshape(train_ratio0_f1,(44,1)), np.reshape(train_ratio1_f1,(44,1)),
                               np.reshape(train_ratio2_f1,(44,1)),np.reshape(train_ratio3_f1,(44,1)), np.reshape(train_ratio4_f1,(44,1)),
                               np.reshape(train_ratio5_f1,(44,1)),np.reshape(train_ratio6_f1,(44,1)), np.reshape(train_ratio7_f1,(44,1)),
                               np.reshape(train_ratio8_f1,(44,1)),np.reshape(train_bs_f1,(44,1))],axis=1)
train_acc = np.concatenate([np.reshape(train_raw_acc,(44,1)), np.reshape(train_ratio0_acc,(44,1)), np.reshape(train_ratio1_acc,(44,1)),
                               np.reshape(train_ratio2_acc,(44,1)),np.reshape(train_ratio3_acc,(44,1)), np.reshape(train_ratio4_acc,(44,1)),
                               np.reshape(train_ratio5_acc,(44,1)),np.reshape(train_ratio6_acc,(44,1)), np.reshape(train_ratio7_acc,(44,1)),
                               np.reshape(train_ratio8_acc,(44,1)),np.reshape(train_bs_acc,(44,1))],axis=1)

train_recall = pd.DataFrame(train_recall)
filename = 'train_recall_ratio.csv'
train_recall.to_csv(filename)

train_precision = pd.DataFrame(train_precision)
filename = 'train_precision_ratio.csv'
train_precision.to_csv(filename)

train_f1 = pd.DataFrame(train_f1)
filename = 'train_f1_ratio.csv'
train_f1.to_csv(filename)

train_acc = pd.DataFrame(train_acc)
filename = 'train_acc_ratio.csv'
train_acc.to_csv(filename)

############## Test ##################
# Raw
test_raw_recall = list()
test_raw_precision = list()
test_raw_f1 = list()
test_raw_acc = list()

path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_raw_recall.append(recall)
    test_raw_precision.append(precision)
    test_raw_f1.append(f1)
    test_raw_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_raw_recall.append(recall)
    test_raw_precision.append(precision)
    test_raw_f1.append(f1)
    test_raw_acc.append(acc)

# Ratio0
test_ratio0_recall = list()
test_ratio0_precision = list()
test_ratio0_f1 = list()
test_ratio0_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio0_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio0_recall.append(recall)
    test_ratio0_precision.append(precision)
    test_ratio0_f1.append(f1)
    test_ratio0_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio0_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio0_recall.append(recall)
    test_ratio0_precision.append(precision)
    test_ratio0_f1.append(f1)
    test_ratio0_acc.append(acc)

# Ratio1
test_ratio1_recall = list()
test_ratio1_precision = list()
test_ratio1_f1 = list()
test_ratio1_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio1_recall.append(recall)
    test_ratio1_precision.append(precision)
    test_ratio1_f1.append(f1)
    test_ratio1_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio1_recall.append(recall)
    test_ratio1_precision.append(precision)
    test_ratio1_f1.append(f1)
    test_ratio1_acc.append(acc)

# Ratio2
test_ratio2_recall = list()
test_ratio2_precision = list()
test_ratio2_f1 = list()
test_ratio2_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio2_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio2_recall.append(recall)
    test_ratio2_precision.append(precision)
    test_ratio2_f1.append(f1)
    test_ratio2_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio2_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio2_recall.append(recall)
    test_ratio2_precision.append(precision)
    test_ratio2_f1.append(f1)
    test_ratio2_acc.append(acc)

# Ratio3
test_ratio3_recall = list()
test_ratio3_precision = list()
test_ratio3_f1 = list()
test_ratio3_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio3_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio3_recall.append(recall)
    test_ratio3_precision.append(precision)
    test_ratio3_f1.append(f1)
    test_ratio3_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio3_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio3_recall.append(recall)
    test_ratio3_precision.append(precision)
    test_ratio3_f1.append(f1)
    test_ratio3_acc.append(acc)

# Ratio4
test_ratio4_recall = list()
test_ratio4_precision = list()
test_ratio4_f1 = list()
test_ratio4_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio4_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio4_recall.append(recall)
    test_ratio4_precision.append(precision)
    test_ratio4_f1.append(f1)
    test_ratio4_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio4_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio4_recall.append(recall)
    test_ratio4_precision.append(precision)
    test_ratio4_f1.append(f1)
    test_ratio4_acc.append(acc)

# Ratio5
test_ratio5_recall = list()
test_ratio5_precision = list()
test_ratio5_f1 = list()
test_ratio5_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio5_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio5_recall.append(recall)
    test_ratio5_precision.append(precision)
    test_ratio5_f1.append(f1)
    test_ratio5_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio5_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio5_recall.append(recall)
    test_ratio5_precision.append(precision)
    test_ratio5_f1.append(f1)
    test_ratio5_acc.append(acc)

# Ratio6
test_ratio6_recall = list()
test_ratio6_precision = list()
test_ratio6_f1 = list()
test_ratio6_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio6_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio6_recall.append(recall)
    test_ratio6_precision.append(precision)
    test_ratio6_f1.append(f1)
    test_ratio6_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio6_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio6_recall.append(recall)
    test_ratio6_precision.append(precision)
    test_ratio6_f1.append(f1)
    test_ratio6_acc.append(acc)

# Ratio7
test_ratio7_recall = list()
test_ratio7_precision = list()
test_ratio7_f1 = list()
test_ratio7_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio7_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio7_recall.append(recall)
    test_ratio7_precision.append(precision)
    test_ratio7_f1.append(f1)
    test_ratio7_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio7_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio7_recall.append(recall)
    test_ratio7_precision.append(precision)
    test_ratio7_f1.append(f1)
    test_ratio7_acc.append(acc)

# Ratio8
test_ratio8_recall = list()
test_ratio8_precision = list()
test_ratio8_f1 = list()
test_ratio8_acc = list()
path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_bsmote_ratio8_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio8_recall.append(recall)
    test_ratio8_precision.append(precision)
    test_ratio8_f1.append(f1)
    test_ratio8_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_BS_SVM_bsmote_ratio8_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_ratio8_recall.append(recall)
    test_ratio8_precision.append(precision)
    test_ratio8_f1.append(f1)
    test_ratio8_acc.append(acc)

# Full bsmote
test_bs_recall = list()
test_bs_precision = list()
test_bs_f1 = list()
test_bs_acc = list()
path2= 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/'
for isub in range(31,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_bs_recall.append(recall)
    test_bs_precision.append(precision)
    test_bs_f1.append(f1)
    test_bs_acc.append(acc)

for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    recall = calculate_recall(result)
    precision = calculate_precision(result)
    f1 = calculate_f1_score(result)
    acc = calculate_acc(result)
    test_bs_recall.append(recall)
    test_bs_precision.append(precision)
    test_bs_f1.append(f1)
    test_bs_acc.append(acc)

test_recall = np.concatenate([np.reshape(test_raw_recall,(44,1)), np.reshape(test_ratio0_recall,(44,1)), np.reshape(test_ratio1_recall,(44,1)),
                               np.reshape(test_ratio2_recall,(44,1)),np.reshape(test_ratio3_recall,(44,1)), np.reshape(test_ratio4_recall,(44,1)),
                               np.reshape(test_ratio5_recall,(44,1)),np.reshape(test_ratio6_recall,(44,1)), np.reshape(test_ratio7_recall,(44,1)),
                               np.reshape(test_ratio8_recall,(44,1)),np.reshape(test_bs_recall,(44,1))],axis=1)
test_precision = np.concatenate([np.reshape(test_raw_precision,(44,1)), np.reshape(test_ratio0_precision,(44,1)), np.reshape(test_ratio1_precision,(44,1)),
                               np.reshape(test_ratio2_precision,(44,1)),np.reshape(test_ratio3_precision,(44,1)), np.reshape(test_ratio4_precision,(44,1)),
                               np.reshape(test_ratio5_precision,(44,1)),np.reshape(test_ratio6_precision,(44,1)), np.reshape(test_ratio7_precision,(44,1)),
                               np.reshape(test_ratio8_precision,(44,1)),np.reshape(test_bs_precision,(44,1))],axis=1)
test_f1 = np.concatenate([np.reshape(test_raw_f1,(44,1)), np.reshape(test_ratio0_f1,(44,1)), np.reshape(test_ratio1_f1,(44,1)),
                               np.reshape(test_ratio2_f1,(44,1)),np.reshape(test_ratio3_f1,(44,1)), np.reshape(test_ratio4_f1,(44,1)),
                               np.reshape(test_ratio5_f1,(44,1)),np.reshape(test_ratio6_f1,(44,1)), np.reshape(test_ratio7_f1,(44,1)),
                               np.reshape(test_ratio8_f1,(44,1)),np.reshape(test_bs_f1,(44,1))],axis=1)
test_acc = np.concatenate([np.reshape(test_raw_acc,(44,1)), np.reshape(test_ratio0_acc,(44,1)), np.reshape(test_ratio1_acc,(44,1)),
                               np.reshape(test_ratio2_acc,(44,1)),np.reshape(test_ratio3_acc,(44,1)), np.reshape(test_ratio4_acc,(44,1)),
                               np.reshape(test_ratio5_acc,(44,1)),np.reshape(test_ratio6_acc,(44,1)), np.reshape(test_ratio7_acc,(44,1)),
                               np.reshape(test_ratio8_acc,(44,1)),np.reshape(test_bs_acc,(44,1))],axis=1)


file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/SVM result/P300_Result_NO.csv'
raw_result = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio0.csv'
ratio0 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio1.csv'
ratio1 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio2.csv'
ratio2 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio3.csv'
ratio3 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio4.csv'
ratio4 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio5.csv'
ratio5 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio6.csv'
ratio6 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio7.csv'
ratio7 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/NO/ratio_result/P300_Result_SVM_borderline_smote_ratio8.csv'
ratio8 = pd.read_csv(file, header=0, index_col=0)
file = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CNN Full ch & NO result/B-SMOTE/P300_Result_SVM_borderline_smote.csv'
bs_result = pd.read_csv(file, header=0, index_col=0)

bci_acc = np.concatenate([raw_result, ratio0, ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7, ratio8, bs_result],axis=1)

test_recall = pd.DataFrame(test_recall)
filename = 'test_recall_ratio.csv'
test_recall.to_csv(filename)

test_precision = pd.DataFrame(test_precision)
filename = 'test_precision_ratio.csv'
test_precision.to_csv(filename)

test_f1 = pd.DataFrame(test_f1)
filename = 'test_f1_ratio.csv'
test_f1.to_csv(filename)

test_acc = pd.DataFrame(test_acc)
filename = 'test_acc_ratio.csv'
test_acc.to_csv(filename)

bci_acc = pd.DataFrame(bci_acc)
filename = 'bci_acc_ratio.csv'
bci_acc.to_csv(filename)