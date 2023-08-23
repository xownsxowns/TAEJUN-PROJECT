## P300 Classification
## NO

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import *
from sklearn.metrics import accuracy_score

total_acc = list()
train_acc = list()

np.random.seed(0)

for isub in range(30,60):
    sam = RandomOverSampler(random_state=5)
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 125
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    # baseline correction

    for i in range(ntrain):
        baseline = data['ERP'][:,0:100,data['target'][i][0]-1,i]
        baseline_mean = np.mean(baseline,axis=1,keepdims=True)
        baseline_std = np.std(baseline,axis=1,keepdims=True)
        target = data['ERP'][:,225:350,data['target'][i][0]-1,i]
        target = (target-baseline_mean)
        tar_data.append(target)
        tar_label.append(1)

        for j in range(4):
            if j == (data['target'][i][0]-1):
                continue
            else:
                nontar = data['ERP'][:,225:350,j,i]
                nontar = (nontar-baseline_mean)
                nontar_data.append(nontar)
                nontar_label.append(0)

    tar_data = np.reshape(tar_data,(ntrain,nlen,nch))
    nontar_data = np.reshape(nontar_data,((ntrain*3),nlen,nch))

    ## Peak-amplitude
    # data point from 225 to 350
    # tar_data shape = (50, 125, 28)
    # ntar_data shape = (150, 125, 28)
    local_peak_tar = np.ones((np.shape(tar_data)[0],np.shape(tar_data)[2]))
    local_peak_latency_tar = np.ones((np.shape(tar_data)[0],np.shape(tar_data)[2]))
    mean_amp_tar = np.ones((np.shape(tar_data)[0],np.shape(tar_data)[2]))
    for ich in range(np.shape(tar_data)[2]):
        for itrial in range(np.shape(tar_data)[0]):
            sorted_data, sorted_index = np.sort(tar_data[itrial,:,ich])[::-1], np.argsort(-tar_data[itrial,:,ich])
            tar = tar_data[itrial,:,ich]
            mean_amp_tar[itrial,ich] = np.mean(tar)
            for i in range(np.shape(tar_data)[1]):
                if sorted_index[i] == np.shape(tar_data)[1]-1 or sorted_index[i] == 0:
                    continue
                elif tar[sorted_index[i]] > tar[sorted_index[i]-1] and tar[sorted_index[i]] > tar[sorted_index[i]+1]:
                    local_peak_tar[itrial,ich] = tar[sorted_index[i]]
                    local_peak_latency_tar[itrial,ich] = sorted_index[i]
                    break

    local_peak_ntar = np.ones((np.shape(nontar_data)[0],np.shape(nontar_data)[2]))
    local_peak_latency_ntar = np.ones((np.shape(nontar_data)[0],np.shape(nontar_data)[2]))
    mean_amp_ntar = np.ones((np.shape(nontar_data)[0],np.shape(nontar_data)[2]))
    for ich in range(np.shape(nontar_data)[2]):
        for itrial in range(np.shape(nontar_data)[0]):
            sorted_data, sorted_index = np.sort(nontar_data[itrial,:,ich])[::-1], np.argsort(-nontar_data[itrial,:,ich])
            ntar = nontar_data[itrial,:,ich]
            mean_amp_ntar[itrial, ich] = np.mean(ntar)
            for i in range(np.shape(nontar_data)[1]):
                if sorted_index[i] == np.shape(nontar_data)[1]-1 or sorted_index[i] == 0:
                    continue
                elif ntar[sorted_index[i]] > ntar[sorted_index[i]-1] and ntar[sorted_index[i]] > ntar[sorted_index[i]+1]:
                    local_peak_ntar[itrial,ich] = ntar[sorted_index[i]]
                    local_peak_latency_ntar[itrial,ich] = sorted_index[i]
                    break

    total_tar_data = np.concatenate((local_peak_tar, local_peak_latency_tar, mean_amp_tar), axis=1)
    total_ntar_data = np.concatenate((local_peak_ntar, local_peak_latency_ntar, mean_amp_ntar), axis=1)
    train_data = np.concatenate((total_tar_data, total_ntar_data))
    train_label = np.concatenate((tar_label, nontar_label))
    data_res, y_res = sam.fit_resample(train_data, train_label)

    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(data_res, y_res)

    train_acc.append(clf.score(data_res, y_res))
    print(train_acc)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    for ii in range(ntest):
        baseline_test = data2['ERP'][:,0:100,:,ii]
        baseline_test_mean = np.mean(baseline_test,axis=1,keepdims=True)
        baseline_test_std = np.std(baseline_test,axis=1,keepdims=True)
        test = data2['ERP'][:, 225:350, :, ii]
        test = (test-baseline_test_mean)
        total_prob = list()
        for j in range(4):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            local_peak_test = np.ones((4, np.shape(test_data)[2]))
            local_peak_latency_test = np.ones((4, np.shape(test_data)[2]))
            mean_amp_test = np.ones((4, np.shape(test_data)[2]))
            for ich in range(np.shape(test_data)[2]):
                for itrial in range(np.shape(test_data)[0]):
                    sorted_data, sorted_index = np.sort(test_data[itrial, :, ich])[::-1], np.argsort(
                        -test_data[itrial, :, ich])
                    testt = test_data[itrial, :, ich]
                    mean_amp_test[itrial,ich] = np.mean(testt)
                    for i in range(np.shape(test_data)[1]):
                        if sorted_index[i] == np.shape(test_data)[1] - 1 or sorted_index[i] == 0:
                            continue
                        elif testt[sorted_index[i]] > testt[sorted_index[i] - 1] and testt[sorted_index[i]] > testt[
                            sorted_index[i] + 1]:
                            local_peak_test[itrial, ich] = testt[sorted_index[i]]
                            local_peak_latency_test[itrial, ich] = sorted_index[i]
                            break
            total_test_data = np.concatenate((local_peak_test, local_peak_latency_test, mean_amp_test), axis=1)
            prob = clf.predict_proba(total_test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmin(total_prob)
        if data2['target'][ii][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))

# BS has 6 icons
for isub in range(14):
    sam = RandomOverSampler(random_state=5)
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 125
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    for i in range(ntrain):
        baseline = data['ERP'][:,0:100,data['target'][i][0]-1,i]
        baseline_mean = np.mean(baseline,axis=1,keepdims=True)
        baseline_std = np.std(baseline,axis=1,keepdims=True)
        target = data['ERP'][:,225:350,data['target'][i][0]-1,i]
        target = (target-baseline_mean)
        tar_data.append(target)
        tar_label.append(1)

        for j in range(6):
            if j == (data['target'][i][0]-1):
                continue
            else:
                nontar = data['ERP'][:,225:350,j,i]
                nontar = (nontar-baseline_mean)
                nontar_data.append(nontar)
                nontar_label.append(0)

    tar_data = np.reshape(tar_data,(ntrain,nlen,nch))
    nontar_data = np.reshape(nontar_data,((ntrain*5),nlen,nch))

    ## Peak-amplitude
    # data point from 225 to 350
    # tar_data shape = (50, 125, 28)
    # ntar_data shape = (150, 125, 28)
    local_peak_tar = np.ones((np.shape(tar_data)[0],np.shape(tar_data)[2]))
    local_peak_latency_tar = np.ones((np.shape(tar_data)[0],np.shape(tar_data)[2]))
    mean_amp_tar = np.ones((np.shape(tar_data)[0],np.shape(tar_data)[2]))
    for ich in range(np.shape(tar_data)[2]):
        for itrial in range(np.shape(tar_data)[0]):
            sorted_data, sorted_index = np.sort(tar_data[itrial,:,ich])[::-1], np.argsort(-tar_data[itrial,:,ich])
            tar = tar_data[itrial,:,ich]
            mean_amp_tar[itrial, ich] = np.mean(tar)
            for i in range(np.shape(tar_data)[1]):
                if sorted_index[i] == np.shape(tar_data)[1]-1 or sorted_index[i] == 0:
                    continue
                elif tar[sorted_index[i]] > tar[sorted_index[i]-1] and tar[sorted_index[i]] > tar[sorted_index[i]+1]:
                    local_peak_tar[itrial,ich] = tar[sorted_index[i]]
                    local_peak_latency_tar[itrial, ich] = sorted_index[i]
                    break

    local_peak_ntar = np.ones((np.shape(nontar_data)[0],np.shape(nontar_data)[2]))
    local_peak_latency_ntar = np.ones((np.shape(nontar_data)[0],np.shape(nontar_data)[2]))
    mean_amp_ntar = np.ones((np.shape(nontar_data)[0],np.shape(nontar_data)[2]))
    for ich in range(np.shape(nontar_data)[2]):
        for itrial in range(np.shape(nontar_data)[0]):
            sorted_data, sorted_index = np.sort(nontar_data[itrial,:,ich])[::-1], np.argsort(-nontar_data[itrial,:,ich])
            ntar = nontar_data[itrial,:,ich]
            mean_amp_ntar[itrial,ich] = np.mean(ntar)
            for i in range(np.shape(nontar_data)[1]):
                if sorted_index[i] == np.shape(nontar_data)[1]-1 or sorted_index[i] == 0:
                    continue
                elif ntar[sorted_index[i]] > ntar[sorted_index[i]-1] and ntar[sorted_index[i]] > ntar[sorted_index[i]+1]:
                    local_peak_ntar[itrial,ich] = ntar[sorted_index[i]]
                    local_peak_latency_ntar[itrial,ich] = sorted_index[i]
                    break

    total_tar_data = np.concatenate((local_peak_tar, local_peak_latency_tar, mean_amp_tar), axis=1)
    total_ntar_data = np.concatenate((local_peak_ntar, local_peak_latency_ntar, mean_amp_ntar), axis=1)
    train_data = np.concatenate((total_tar_data, total_ntar_data))
    train_label = np.concatenate((tar_label, nontar_label))
    data_res, y_res = sam.fit_resample(train_data, train_label)

    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(data_res, y_res)

    train_acc.append(clf.score(data_res, y_res))
    print(train_acc)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    for ii in range(ntest):
        baseline_test = data2['ERP'][:,0:100,:,ii]
        baseline_test_mean = np.mean(baseline_test,axis=1,keepdims=True)
        baseline_test_std = np.std(baseline_test,axis=1,keepdims=True)
        test = data2['ERP'][:, 225:350, :, ii]
        test = (test-baseline_test_mean)
        total_prob = list()
        for j in range(6):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            local_peak_test = np.ones((6, np.shape(test_data)[2]))
            local_peak_latency_test = np.ones((6, np.shape(test_data)[2]))
            mean_amp_test = np.ones((6, np.shape(test_data)[2]))
            for ich in range(np.shape(test_data)[2]):
                for itrial in range(np.shape(test_data)[0]):
                    sorted_data, sorted_index = np.sort(test_data[itrial, :, ich])[::-1], np.argsort(
                        -test_data[itrial, :, ich])
                    testt = test_data[itrial, :, ich]
                    mean_amp_test[itrial, ich] = np.mean(testt)
                    for i in range(np.shape(test_data)[1]):
                        if sorted_index[i] == np.shape(test_data)[1] - 1 or sorted_index[i] == 0:
                            continue
                        elif testt[sorted_index[i]] > testt[sorted_index[i] - 1] and testt[sorted_index[i]] > testt[
                            sorted_index[i] + 1]:
                            local_peak_test[itrial, ich] = testt[sorted_index[i]]
                            local_peak_latency_test[itrial, ich] = sorted_index[i]
                            break
            total_test_data = np.concatenate((local_peak_test, local_peak_latency_test, mean_amp_test), axis=1)
            prob = clf.predict_proba(total_test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmin(total_prob)
        if data2['target'][ii][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)

df = pd.DataFrame(total_acc)
filename = 'P300_Result_ROS_raw.csv'
df.to_csv(filename)

df2 = pd.DataFrame(train_acc)
filename = 'P300_Result_ROS_raw_train.csv'
df2.to_csv(filename)