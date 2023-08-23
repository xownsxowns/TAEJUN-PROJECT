
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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

total_acc = list()

for isub in range(11):
    print(isub+1)
    sm = BorderlineSMOTE(random_state=20)
    path = '/Volumes/UNTITLED2/eeg/epoch/sub'+str(isub+1)+'_train_data.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    for i in range(ntrain):
        target = data['ERP'][:,150:,data['target'][0][i]-1,i]
        tar_data.append(target)
        tar_label.append(1)

        for j in range(6):
            if j == (data['target'][0][i]-1):
                continue
            else:
                nontar_data.append(data['ERP'][:,150:,j,i])
                nontar_label.append(0)

    tar_data = np.reshape(tar_data,(ntrain,nlen,nch))
    nontar_data = np.reshape(nontar_data,((ntrain*5),nlen,nch))

    train_data = np.concatenate((tar_data, nontar_data))
    train_label = np.concatenate((tar_label, nontar_label))

    ori_shape = train_data.shape
    reshape_data = np.reshape(train_data, (train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
    data_res, y_res = sm.fit_resample(reshape_data, train_label)
    data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(data_res.shape[1]):
        scalers[i] = StandardScaler()
        data_res[:, i, :] = scalers[i].fit_transform(data_res[:, i, :])

    new_train_data = data_res.reshape((data_res.shape[0], (data_res.shape[1] * data_res.shape[2])))
    clf = SVC(probability=True, kernel='sigmoid')
    clf.fit(new_train_data, y_res)

    ## Test
    path = '/Volumes/UNTITLED2/eeg/epoch/sub'+str(isub+1)+'_test_data.mat'
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
        if data2['target'][0][i] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)

df = pd.DataFrame(total_acc)
filename = 'P300_Result_SVM_borderlinesmote_sigmoid2.csv'
df.to_csv(filename)