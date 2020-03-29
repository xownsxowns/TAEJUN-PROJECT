import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CNN_full ch result/confusion matrix/'
total_sum_cnn = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_CNN_full_ch_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn = total_sum_cnn + result

total_sum_cnn = total_sum_cnn.astype(int)
plt.figure(figsize = (10,7))
sns.heatmap(total_sum_cnn, annot=True, fmt="d")
plt.title('Confusion matrix (CNN full ch, Doorlock)')
plt.show()

path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/NO result/confusion matrix/'
total_sum_svm = np.zeros((2,2))
for isub in range(31,46):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm = total_sum_svm + result

total_sum_svm = total_sum_svm.astype(int)
plt.figure(figsize = (10,7))
sns.heatmap(total_sum_svm, annot=True, fmt="d")
plt.title('Confusion matrix (NO, Doorlock)')
plt.show()

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CNN_full ch result/confusion matrix/'
total_sum_cnn = np.zeros((2,2))
for isub in range(46,61):
    file = path + 'P300_Result_CNN_full_ch_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn = total_sum_cnn + result

total_sum_cnn = total_sum_cnn.astype(int)
plt.figure(figsize = (10,7))
sns.heatmap(total_sum_cnn, annot=True, fmt="d")
plt.title('Confusion matrix (CNN full ch, Lamp)')
plt.show()

path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/NO result/confusion matrix/'
total_sum_svm = np.zeros((2,2))
for isub in range(46,61):
    file = path2 + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm = total_sum_svm + result

total_sum_svm = total_sum_svm.astype(int)
plt.figure(figsize = (10,7))
sns.heatmap(total_sum_svm, annot=True, fmt="d")
plt.title('Confusion matrix (NO, Lamp)')
plt.show()

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CNN_full ch result/confusion matrix/'
total_sum_cnn = np.zeros((2,2))
for isub in range(1,15):
    file = path + 'P300_Result_CNN_BS_full_ch_total_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn = total_sum_cnn + result

total_sum_cnn = total_sum_cnn.astype(int)
plt.figure(figsize = (10,7))
sns.heatmap(total_sum_cnn, annot=True, fmt="d")
plt.title('Confusion matrix (CNN full ch, Bluetooth speaker)')
plt.show()

path2 = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/NO result/confusion matrix/'
total_sum_svm = np.zeros((2,2))
for isub in range(1,15):
    file = path2 + 'P300_Result_SVM_BS_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_svm = total_sum_svm + result

total_sum_svm = total_sum_svm.astype(int)
plt.figure(figsize = (10,7))
sns.heatmap(total_sum_svm, annot=True, fmt="d")
plt.title('Confusion matrix (NO, Bluetooth speaker)')
plt.show()