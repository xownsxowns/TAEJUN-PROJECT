## CNN
## P300 Classification

# Epoch Sub31 ~ Sub45: Doorlock
# Epoch Sub46 ~ Sub60: Lamp
# Epoch BS Sub 1 ~Sub 14: Bluetooth speaker

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
import random
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import collections
from keras_radam import RAdam

total_acc = list()
target_data = list()
nontarget_data = list()
target_label = list()
nontarget_label = list()

## extracting overlap channel
for isub in range(30,60):
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)
print('extracting overlap channel end')


for isub in range(30,60):
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
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

    ## channel selection


    # channel 개수 같아야돼 (차원수 확인!!)
    target_data.append(tar_data)
    nontarget_data.append(nontar_data)
    target_label.append(tar_label)
    nontarget_label.append(nontar_label)

target_data = np.array(target_data)
target_label = np.array(target_label)
nontarget_data = np.array(nontarget_data)
nontarget_label = np.array(nontarget_label)

for isub in range(np.shape(target_data)[0]):
    if isub == 0:
        total_tar = target_data[isub,:,:,:]
        total_tar_label = target_label[isub,:]
    else:
        total_tar = np.concatenate((total_tar,target_data[isub,:,:,:]),axis=0)
        total_tar_label = np.concatenate((total_tar_label,target_label[isub,:]),axis=0)

for isub in range(np.shape(nontarget_data)[0]):
    if isub == 0:
        total_nontar = nontarget_data[isub, :, :, :]
        total_nontar_label = nontarget_label[isub, :]
    else:
        total_nontar = np.concatenate((total_nontar, nontarget_data[isub, :, :, :]), axis=0)
        total_nontar_label = np.concatenate((total_nontar_label, nontarget_label[isub, :]), axis=0)



    train_vali_data = np.concatenate((tar_data, nontar_data))
    train_vali_label = np.concatenate((tar_label, nontar_label))

    train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.15, random_state=42)

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
        vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10 , input_shape=(nlen, nch), activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(RAdam(), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    early_stopping = EarlyStopping(patience=10)
    model.fit(train_data, train_label, epochs=200, batch_size=20, validation_data=(vali_data, vali_label), callbacks=[early_stopping])

    ## Test
    # path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_test.mat'
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
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[i].transform(test_data[:, k, :])
            prob = model.predict_proba(test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)

df = pd.DataFrame(total_acc)
filename = 'P300_Result_CNN_TL.csv'
df.to_csv(filename)