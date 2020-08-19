
## P300 Classification
## CNN feature extraction

# Epoch Sub1 ~ Sub30: TV
# Epoch Sub31 ~ Sub45: Doorlock
# Epoch Sub46 ~ Sub60: Lamp
# Epoch BS Sub 1 ~Sub45: Bluetooth speaker

# 1. Preprocessing
#  1) 0.5Hz highpass filter (FIR)
#  2) Bad channel rejection (1Hz lowpass filter , 2nd order Butter. , Corr. coeff < 0.4 , 70 % above)
#  3) Common average re-reference
#  4) 50Hz lowpass filter (FIR)
#  5) Artifact subspace reconstruction (cutoff: 10)
#
# 2. Data
#    ERP : [channel x time x stimulus type x block] (training: 50 block, test: 30 block)
#    target : [block x 1] target stimulus of each block

from scipy import io, signal
import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import *
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import *

# parameter setting
total_acc = list()
train_score = list()
train_score_prob = list()
np.random.seed(0)

for isub in range(30,60):
    ncr = NeighbourhoodCleaningRule()
    print(isub)
    path = 'D:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
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
    data_res, y_res = ncr.fit_resample(reshape_data, train_label)

    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(data_res, y_res)

    ## prob로 하지 않고 그냥 predict로 했을 때
    training_score = accuracy_score(y_res, clf.predict(data_res))
    train_score.append(training_score)

    ## prob으로 했을 때
    tarr = train_data[:50,:,:]
    ntarr = train_data[50:,:,:]
    corr_train_ans = 0

    for aa in range(50):
        tarr_data = tarr[aa,:,:]
        ntarr_data = ntarr[3*aa:3*(aa+1),:,:]
        tarr_data = np.reshape(tarr_data, (tarr_data.shape[0]*tarr_data.shape[1]))
        tarr_data = np.expand_dims(tarr_data, axis=0)
        ntarr_data = np.reshape(ntarr_data, (ntarr_data.shape[0],(ntarr_data.shape[1]*ntarr_data.shape[2])))
        ttrain_data = np.concatenate((tarr_data, ntarr_data))
        probb = clf.predict_proba(ttrain_data)
        predicted_tar = np.argmin(probb[:,0])
        if predicted_tar == 0:
            corr_train_ans += 1
    train_score_prob.append((corr_train_ans/50)*100)

    ## Test
    path = 'D:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        total_prob = list()
        for j in range(4):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            new_test_data = test_data.reshape((test_data.shape[0], (test_data.shape[1] * test_data.shape[2])))
            prob = clf.predict_proba(new_test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmin(total_prob)
        if data2['target'][i][0] == (predicted_label + 1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))

for isub in range(14):
    ncr = NeighbourhoodCleaningRule()
    print(isub)
    path = 'D:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
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
    data_res, y_res = ncr.fit_resample(reshape_data, train_label)

    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(data_res, y_res)

    ## prob로 하지 않고 그냥 predict로 했을 때
    training_score = accuracy_score(y_res, clf.predict(data_res))
    train_score.append(training_score)

    ## prob으로 했을 때
    tarr = train_data[:50, :, :]
    ntarr = train_data[50:, :, :]
    corr_train_ans = 0

    for aa in range(50):
        tarr_data = tarr[aa,:,:]
        ntarr_data = ntarr[5*aa:5*(aa+1),:,:]
        tarr_data = np.reshape(tarr_data, (tarr_data.shape[0]*tarr_data.shape[1]))
        tarr_data = np.expand_dims(tarr_data, axis=0)
        ntarr_data = np.reshape(ntarr_data, (ntarr_data.shape[0],(ntarr_data.shape[1]*ntarr_data.shape[2])))
        ttrain_data = np.concatenate((tarr_data, ntarr_data))
        probb = clf.predict_proba(ttrain_data)
        predicted_tar = np.argmin(probb[:,0])
        if predicted_tar == 0:
            corr_train_ans += 1
    train_score_prob.append((corr_train_ans/50)*100)
    ## Test
    path = 'D:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        total_prob = list()
        for j in range(6):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            new_test_data = test_data.reshape((test_data.shape[0], (test_data.shape[1] * test_data.shape[2])))
            prob = clf.predict_proba(new_test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmin(total_prob)
        if data2['target'][i][0] == (predicted_label + 1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))

df = pd.DataFrame(total_acc)
filename = 'P300_Result_SVM_ncr.csv'
df.to_csv(filename)

df2 = pd.DataFrame(train_score)
filename = 'P300_Result_SVM_ncr_trainscore.csv'
df2.to_csv(filename)

df3 = pd.DataFrame(train_score_prob)
filename = 'P300_Result_SVM_ncr_trainscore_prob.csv'
df3.to_csv(filename)