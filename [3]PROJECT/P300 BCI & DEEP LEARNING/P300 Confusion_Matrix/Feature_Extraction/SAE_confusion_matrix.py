
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

from scipy import io
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model
import gc
import keras.backend as K
from sklearn.metrics import confusion_matrix

np.random.seed(0)
total_acc = list()

for isub in range(30,60):
    print(isub)
    model_name = 'E:/[9] 졸업논문/model/feature_extraction/SAE/model_SAE2_train' + str(
        isub + 1) + '.h5'
    model = load_model(model_name)

    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
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

    train_vali_data = np.concatenate((tar_data, nontar_data))
    train_vali_label = np.concatenate((tar_label, nontar_label))

    train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.10, random_state=42)

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
        vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'

    data2 = io.loadmat(path)
    corr_ans = 0
    ntest = np.shape(data2['ERP'])[3]

    total_label = list()
    total_class = list()

    for i in range(ntest):
        test = data2['ERP'][:,150:,:,i]
        total_prob = list()
        for j in range(4):
            test_data = test[:, :, j]
            test_data = np.reshape(test_data, (1, nlen, nch))
            for k in range(test_data.shape[1]):
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
            predicted_class = model.predict_classes(test_data)
            total_class.append(predicted_class[0][0])
            if j == (data2['target'][i][0] - 1):
                total_label.append(1)
            else:
                total_label.append(0)

    confusion_mat = confusion_matrix(total_label, total_class)
    df = pd.DataFrame(confusion_mat)
    filename = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/' \
               '/P300_Result_SAE_confusion_' + str(isub + 1) + '.csv'
    df.to_csv(filename)

    K.clear_session()
    gc.collect()
    del model


for isub in range(14):
    print(isub)
    model_name = 'E:/[9] 졸업논문/model/feature_extraction/SAE/model_BS_SAE2_train' + str(
        isub + 1) + '.h5'
    model = load_model(model_name)

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

    train_vali_data = np.concatenate((tar_data, nontar_data))
    train_vali_label = np.concatenate((tar_label, nontar_label))

    train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.10, random_state=42)

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
        vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])

    total_label = list()
    total_class = list()

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
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
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
            predicted_class = model.predict_classes(test_data)
            total_class.append(predicted_class[0][0])
            if j == (data2['target'][i][0] - 1):
                total_label.append(1)
            else:
                total_label.append(0)

    confusion_mat = confusion_matrix(total_label, total_class)
    df = pd.DataFrame(confusion_mat)
    filename = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/' \
               '/P300_Result_BS_SAE_confusion_' + str(isub + 1) + '.csv'
    df.to_csv(filename)

    K.clear_session()
    gc.collect()
    del model