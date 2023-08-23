
## P300 Classification
## CNN

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
import random
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

total_acc = list()

## extracting overlap channel
from scipy import io
path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/chlist_light.mat'
data = io.loadmat(path)

# 공통인수 뽑고, 각 sub별 채널리스트에서 index받아와서 데이터 가져와
light_ch_list = []
for isub in range(len(data['chlist_light'][0])):
    light_ch_list.append(data['chlist_light'][0][isub][0])

# 공통인수뽑기
elements_in_all = list(set.intersection(*map(set, light_ch_list)))
set_elements = set(elements_in_all)
# sub별 index 뽑기
totalsub_index = []
for isub in range(len(data['chlist_light'][0])):
    list_light = data['chlist_light'][0][isub][0]
    list_index = [i for i,e in enumerate(list_light) if e in set_elements]
    totalsub_index.append(list_index)

for isub in range(45,60):
    print(isub)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)

    ch_ind = totalsub_index[isub - 45]
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

    tar_data = tar_data[:,:,ch_ind]
    nontar_data = nontar_data[:,:,ch_ind]

    train_vali_data = np.concatenate((tar_data, nontar_data))
    train_vali_label = np.concatenate((tar_label, nontar_label))

    train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.15, random_state=42)

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
        vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])

    nnch = np.shape(tar_data)[2]

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10 , input_shape=(nlen, nnch)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=64, kernel_size=10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    early_stopping = EarlyStopping(patience=10)
    model.fit(train_data, train_label, epochs=200, batch_size=20, validation_data=(vali_data, vali_label), callbacks=[early_stopping])

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
            test_data = test_data[:,:,ch_ind]
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            prob = model.predict_proba(test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))

df = pd.DataFrame(total_acc)
filename = 'P300_Result_CNN_BN_ch.csv'
df.to_csv(filename)