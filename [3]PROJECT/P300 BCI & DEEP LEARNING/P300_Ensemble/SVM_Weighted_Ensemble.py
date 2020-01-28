
## P300 Classification
## SVM

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


## validation, early stopping

from scipy import io, signal
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_score1 = list()
train_score2 = list()
train_score3 = list()
bs_train_score1 = list()
bs_train_score2 = list()
bs_train_score3 = list()
bs_train_score4 = list()
bs_train_score5 = list()

total_acc = list()
np.random.seed(0)
random.seed(0)

for isub in range(30,60):
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/UNTITLED2/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
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

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])

    tar_data = train_data[0:50,:,:]
    nontar_data = train_data[50:,:,:]

    # divide nontar_data (3 subset, 150)
    arr = np.arange(150)
    np.random.shuffle(arr)
    subset1_ind = arr[0:50]
    subset2_ind = arr[50:100]
    subset3_ind = arr[100:150]

    ntar_data1 = nontar_data[subset1_ind,:,:]
    ntar_data2 = nontar_data[subset2_ind,:,:]
    ntar_data3 = nontar_data[subset3_ind,:,:]
    ntar_label = np.zeros(50)

    train_data1 = np.concatenate((tar_data, ntar_data1))
    train_label1 = np.concatenate((tar_label, ntar_label))

    train_data2 = np.concatenate((tar_data, ntar_data2))
    train_label2 = np.concatenate((tar_label, ntar_label))

    train_data3 = np.concatenate((tar_data, ntar_data3))
    train_label3 = np.concatenate((tar_label, ntar_label))

    new_train_data1 = train_data1.reshape((train_data1.shape[0], (train_data1.shape[1] * train_data1.shape[2])))
    clf1 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf1.fit(new_train_data1, train_label1)

    ## prob로 하지 않고 그냥 predict로 했을 때
    training_score1 = accuracy_score(train_label1, clf1.predict(new_train_data1))
    train_score1.append(training_score1)

    new_train_data2 = train_data2.reshape((train_data2.shape[0], (train_data2.shape[1] * train_data2.shape[2])))
    clf2 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf2.fit(new_train_data2, train_label2)

    ## prob로 하지 않고 그냥 predict로 했을 때
    training_score2 = accuracy_score(train_label2, clf2.predict(new_train_data2))
    train_score2.append(training_score2)

    new_train_data3 = train_data3.reshape((train_data3.shape[0], (train_data3.shape[1] * train_data3.shape[2])))
    clf3 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf3.fit(new_train_data3, train_label3)

    ## prob로 하지 않고 그냥 predict로 했을 때
    training_score3 = accuracy_score(train_label3, clf3.predict(new_train_data3))
    train_score3.append(training_score3)

    # 정확도 sorting index
    sort_index = np.argsort([training_score1, training_score2, training_score3])

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_test.mat'
    # path = '/Volumes/UNTITLED2/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
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
            prob1 = clf1.predict_proba(new_test_data)
            prob2 = clf2.predict_proba(new_test_data)
            prob3 = clf3.predict_proba(new_test_data)
            tar_prob = [prob1[0][1], prob2[0][1], prob3[0][1]]
            total_prob.append((tar_prob[sort_index[0]]*(1/6)+tar_prob[sort_index[1]]*(2/6)+tar_prob[sort_index[2]]*(3/6)))
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)

# BS has 6 icons
for isub in range(14):
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Users/Taejun/Desktop/현대실무연수자료/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/UNTITLED2/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
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

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])

    tar_data = train_data[0:50,:,:]
    nontar_data = train_data[50:,:,:]

    # divide nontar_data (5 subset, 250)
    arr = np.arange(250)
    np.random.shuffle(arr)
    subset1_ind = arr[0:50]
    subset2_ind = arr[50:100]
    subset3_ind = arr[100:150]
    subset4_ind = arr[150:200]
    subset5_ind = arr[200:250]
    subset6_ind = arr[250:300]

    ntar_data1 = nontar_data[subset1_ind,:,:]
    ntar_data2 = nontar_data[subset2_ind,:,:]
    ntar_data3 = nontar_data[subset3_ind,:,:]
    ntar_data4 = nontar_data[subset4_ind,:,:]
    ntar_data5 = nontar_data[subset5_ind,:,:]
    ntar_label = np.zeros(50)

    train_data1 = np.concatenate((tar_data, ntar_data1))
    train_label1 = np.concatenate((tar_label, ntar_label))

    train_data2 = np.concatenate((tar_data, ntar_data2))
    train_label2 = np.concatenate((tar_label, ntar_label))

    train_data3 = np.concatenate((tar_data, ntar_data3))
    train_label3 = np.concatenate((tar_label, ntar_label))

    train_data4 = np.concatenate((tar_data, ntar_data4))
    train_label4 = np.concatenate((tar_label, ntar_label))

    train_data5 = np.concatenate((tar_data, ntar_data5))
    train_label5 = np.concatenate((tar_label, ntar_label))

    new_train_data1 = train_data1.reshape((train_data1.shape[0], (train_data1.shape[1] * train_data1.shape[2])))
    clf1 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf1.fit(new_train_data1, train_label1)

    ## prob로 하지 않고 그냥 predict로 했을 때
    bs_training_score1 = accuracy_score(train_label1, clf1.predict(new_train_data1))
    bs_train_score1.append(bs_training_score1)

    new_train_data2 = train_data2.reshape((train_data2.shape[0], (train_data2.shape[1] * train_data2.shape[2])))
    clf2 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf2.fit(new_train_data2, train_label2)

    ## prob로 하지 않고 그냥 predict로 했을 때
    bs_training_score2 = accuracy_score(train_label2, clf2.predict(new_train_data2))
    bs_train_score2.append(bs_training_score2)

    new_train_data3 = train_data3.reshape((train_data3.shape[0], (train_data3.shape[1] * train_data3.shape[2])))
    clf3 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf3.fit(new_train_data3, train_label3)

    ## prob로 하지 않고 그냥 predict로 했을 때
    bs_training_score3 = accuracy_score(train_label3, clf3.predict(new_train_data3))
    bs_train_score3.append(bs_training_score3)

    new_train_data4 = train_data4.reshape((train_data4.shape[0], (train_data4.shape[1] * train_data4.shape[2])))
    clf4 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf4.fit(new_train_data4, train_label4)

    ## prob로 하지 않고 그냥 predict로 했을 때
    bs_training_score4 = accuracy_score(train_label4, clf4.predict(new_train_data4))
    bs_train_score4.append(bs_training_score4)

    new_train_data5 = train_data5.reshape((train_data5.shape[0], (train_data5.shape[1] * train_data5.shape[2])))
    clf5 = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf5.fit(new_train_data5, train_label5)

    ## prob로 하지 않고 그냥 predict로 했을 때
    bs_training_score5 = accuracy_score(train_label5, clf5.predict(new_train_data5))
    bs_train_score5.append(bs_training_score5)

    # 정확도 sorting index
    bs_sort_index = np.argsort([bs_training_score1, bs_training_score2, bs_training_score3, bs_training_score4, bs_training_score5])

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
    # path = '/Users/Taejun/Desktop/현대실무연수자료/Epoch_BS/Sub' + str(isub + 1) + '_EP_test.mat'
    # path = '/Volumes/UNTITLED2/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
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
            prob1 = clf1.predict_proba(new_test_data)
            prob2 = clf2.predict_proba(new_test_data)
            prob3 = clf3.predict_proba(new_test_data)
            prob4 = clf4.predict_proba(new_test_data)
            prob5 = clf5.predict_proba(new_test_data)

            tar_prob = [prob1[0][1], prob2[0][1], prob3[0][1], prob4[0][1], prob5[0][1]]
            total_prob.append((tar_prob[bs_sort_index[0]] * (1/15) + tar_prob[bs_sort_index[1]] * (2/15) + tar_prob[
                bs_sort_index[2]] * (3/15) + tar_prob[bs_sort_index[3]]*(4/15) + tar_prob[bs_sort_index[4]]*(5/15)))
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)

df = pd.DataFrame(total_acc)
filename = 'P300_Result_weighted_multisvm.csv'
df.to_csv(filename)
