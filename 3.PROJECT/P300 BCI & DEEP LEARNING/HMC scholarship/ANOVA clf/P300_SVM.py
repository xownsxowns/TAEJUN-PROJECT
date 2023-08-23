
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


## validation, early stopping

from scipy import io, signal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import scipy.stats as stats
import collections

total_acc = list()

for isub in range(30,60):
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
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

    # ANOVA test
    tar_anova = {}
    ntar_anova = {}
    time_ind = {}
    ich_index = list()
    for ich in range(np.shape(tar_data)[2]):
        itime_index = list()
        for itime in range(np.shape(tar_data)[1]):
            tar_group = tar_data[:,itime,ich]
            ntar_group = nontar_data[:,itime,ich]
            F_statistic, pVal = stats.f_oneway(tar_group, ntar_group)
            if pVal < 0.05:
                ich_index.append(ich)
                itime_index.append(itime)
        tar_anova[ich] = tar_data[:,itime_index,ich]
        ntar_anova[ich] = nontar_data[:,itime_index,ich]
        time_ind[ich] = itime_index

    ch_ind = [item for item, count in collections.Counter(ich_index).items() if count > 1]

    after_tar_data = []
    for ich in range(np.shape(ch_ind)[0]):
        if ich == 0:
            after_tar_data = tar_anova[ich]
        else:
            after_tar_data = np.concatenate((after_tar_data, tar_anova[ich]),axis=1)
    after_ntar_data = []
    for ich in range(np.shape(ch_ind)[0]):
        if ich == 0:
            after_ntar_data = ntar_anova[ich]
        else:
            after_ntar_data = np.concatenate((after_ntar_data, ntar_anova[ich]),axis=1)

    train_data = np.concatenate((after_tar_data, after_ntar_data))
    train_label = np.concatenate((tar_label, nontar_label))

    clf = SVC(probability=True, kernel='sigmoid')
    clf.fit(train_data, train_label)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_test.mat'
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        total_prob = list()
        for j in range(4):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            for ich in range(len(ch_ind)):
                dataaa = test_data[:,time_ind[ich],ch_ind[ich]]
                if ich == 0:
                    new_test_data = dataaa
                else:
                    new_test_data = np.concatenate((new_test_data, dataaa), axis=1)
            prob = clf.predict_proba(new_test_data)
            # prob = clf.predict(new_test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmin(total_prob)
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
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
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

    # ANOVA test
    tar_anova = {}
    ntar_anova = {}
    time_ind = {}
    ich_index = list()
    for ich in range(np.shape(tar_data)[2]):
        itime_index = list()
        for itime in range(np.shape(tar_data)[1]):
            tar_group = tar_data[:,itime,ich]
            ntar_group = nontar_data[:,itime,ich]
            F_statistic, pVal = stats.f_oneway(tar_group, ntar_group)
            if pVal < 0.05:
                ich_index.append(ich)
                itime_index.append(itime)
        tar_anova[ich] = tar_data[:,itime_index,ich]
        ntar_anova[ich] = nontar_data[:,itime_index,ich]
        time_ind[ich] = itime_index

    ch_ind = [item for item, count in collections.Counter(ich_index).items() if count > 1]

    after_tar_data = []
    for ich in range(np.shape(ch_ind)[0]):
        if ich == 0:
            after_tar_data = tar_anova[ich]
        else:
            after_tar_data = np.concatenate((after_tar_data, tar_anova[ich]),axis=1)
    after_ntar_data = []
    for ich in range(np.shape(ch_ind)[0]):
        if ich == 0:
            after_ntar_data = ntar_anova[ich]
        else:
            after_ntar_data = np.concatenate((after_ntar_data, ntar_anova[ich]),axis=1)

    train_data = np.concatenate((after_tar_data, after_ntar_data))
    train_label = np.concatenate((tar_label, nontar_label))

    clf = SVC(probability=True, kernel='sigmoid')
    clf.fit(train_data, train_label)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
    # path = '/Users/Taejun/Desktop/현대실무연수자료/Epoch_BS/Sub' + str(isub + 1) + '_EP_test.mat'
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        total_prob = list()
        for j in range(6):
            test_data = test[:,:,j]
            test_data = np.reshape(test_data, (1,nlen,nch))
            for ich in range(len(ch_ind)):
                dataaa = test_data[:,time_ind[ich],ch_ind[ich]]
                if ich == 0:
                    new_test_data = dataaa
                else:
                    new_test_data = np.concatenate((new_test_data, dataaa), axis=1)
            prob = clf.predict_proba(new_test_data)
            # prob = clf.predict(new_test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmin(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)

df = pd.DataFrame(total_acc)
filename = 'P300_Result_SVM_ANOVA.csv'
df.to_csv(filename)