
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
total_acc = list()
train_score = list()
train_score_prob = list()
np.random.seed(0)

sv_ratio_total = list()
sv_target_num = list()
sv_nontarget_num = list()
sv_distance_total = list()
# recall = list()
for isub in range(30,60):
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

    new_train_data = train_data.reshape((train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(new_train_data, train_label)

    ## SVM analysis
    support = clf.support_
    sv_label = train_label[support]
    sv = clf.support_vectors_
    sv_target = (sv_label == 1).sum()
    sv_target_num.append(sv_target)
    sv_nontarget = (sv_label == 0).sum()
    sv_nontarget_num.append(sv_nontarget)
    sv_ratio = sv_target / sv_nontarget
    sv_ratio_total.append(sv_ratio)
    # decision_function: relative distance, 0 is decision boundary
    df = clf.decision_function(new_train_data)
    df_support = df[support]
    df_total = np.mean(np.abs(df_support))
    sv_distance_total.append(df_total)
    # dual_coef = clf.dual_coef_


# BS has 6 icons
for isub in range(14):
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

    new_train_data = train_data.reshape((train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
    clf.fit(new_train_data, train_label)

    ## SVM analysis
    support = clf.support_
    sv_label = train_label[support]
    sv = clf.support_vectors_
    sv_target = (sv_label == 1).sum()
    sv_target_num.append(sv_target)
    sv_nontarget = (sv_label == 0).sum()
    sv_nontarget_num.append(sv_nontarget)
    sv_ratio = sv_target / sv_nontarget
    sv_ratio_total.append(sv_ratio)
    # decision_function: relative distance, 0 is decision boundary
    df = clf.decision_function(new_train_data)
    df_support = df[support]
    df_total = np.mean(np.abs(df_support))
    sv_distance_total.append(df_total)
    # dual_coef = clf.dual_coef_

sv_ratio_total_pd = pd.DataFrame(sv_ratio_total)
sv_target_num_pd = pd.DataFrame(sv_target_num)
sv_nontarget_num_pd = pd.DataFrame(sv_nontarget_num)
sv_distance_total_pd = pd.DataFrame(sv_distance_total)

sv_ratio_total_pd.to_csv('sv_ratio_total.csv')
sv_target_num_pd.to_csv('sv_target_num.csv')
sv_nontarget_num_pd.to_csv('sv_nontarget_num.csv')
sv_distance_total_pd.to_csv('sv_distance_total.csv')

