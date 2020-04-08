from scipy import io, signal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import *

np.random.seed(0)

for isub in range(30,60):
    sm = BorderlineSMOTE(random_state=5)
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    for i in range(ntrain):
        target = data['ERP'][:,150:,data['target'][i][0]-1,i]
        tar_data.append(target)
        tar_label.append(1)

        for j in range(4):
            if j == (data['target'][i][0]-1):
                continue
            else:
                nontar_data.append(data['ERP'][:,150:,j,i])
                nontar_label.append(0)

    tar_data = np.reshape(tar_data,(ntrain,nlen,nch))
    nontar_data = np.reshape(nontar_data,((ntrain*3),nlen,nch))

    train_data = np.concatenate((tar_data, nontar_data))
    train_label = np.concatenate((tar_label, nontar_label))

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])

    ori_shape = train_data.shape
    reshape_data = np.reshape(train_data, (train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    data_res, y_res = sm.fit_resample(reshape_data, train_label)

    new_train_data = train_data.reshape((train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(new_train_data, train_label)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    total_label = list()
    total_class = list()
    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        for j in range(4):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            new_test_data = test_data.reshape((test_data.shape[0], (test_data.shape[1] * test_data.shape[2])))
            predicted_class = clf.predict(new_test_data)
            total_class.append(predicted_class[0])
            if j == (data2['target'][i][0] - 1):
                total_label.append(1)
            else:
                total_label.append(0)

    confusion_mat = confusion_matrix(total_label, total_class)
    df = pd.DataFrame(confusion_mat)
    filename = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling' \
               '/CONFUSION/B-SMOTE/P300_Result_SVM_confusion_' + str(isub+1) + '.csv'
    df.to_csv(filename)


for isub in range(14):
    sm = BorderlineSMOTE(random_state=5)
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    for i in range(ntrain):
        target = data['ERP'][:,150:,data['target'][i][0]-1,i]
        tar_data.append(target)
        tar_label.append(1)

        for j in range(6):
            if j == (data['target'][i][0]-1):
                continue
            else:
                nontar_data.append(data['ERP'][:,150:,j,i])
                nontar_label.append(0)

    tar_data = np.reshape(tar_data,(ntrain,nlen,nch))
    nontar_data = np.reshape(nontar_data,((ntrain*5),nlen,nch))

    train_data = np.concatenate((tar_data, nontar_data))
    train_label = np.concatenate((tar_label, nontar_label))

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])

    ori_shape = train_data.shape
    reshape_data = np.reshape(train_data, (train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    data_res, y_res = sm.fit_resample(reshape_data, train_label)

    new_train_data = train_data.reshape((train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(new_train_data, train_label)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    total_label = list()
    total_class = list()
    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        for j in range(6):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            new_test_data = test_data.reshape((test_data.shape[0], (test_data.shape[1] * test_data.shape[2])))
            predicted_class = clf.predict(new_test_data)
            total_class.append(predicted_class[0])
            if j == (data2['target'][i][0] - 1):
                total_label.append(1)
            else:
                total_label.append(0)

    confusion_mat = confusion_matrix(total_label, total_class)
    df = pd.DataFrame(confusion_mat)
    filename = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE' \
               '/P300_Result_SVM_BS_confusion_' + str(isub+1) + '.csv'
    df.to_csv(filename)
