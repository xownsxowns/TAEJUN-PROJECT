
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
import random
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.optimizers import Adam
from imblearn.over_sampling import *
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
import keras.backend as K

# parameter setting

np.random.seed(0)
random.seed(0)

for repeat_num in range(1,2):
    total_acc = list()
    train_score = list()
    for isub in range(30,60):
        adasyn = BorderlineSMOTE(random_state=5)
        model_name = 'E:/[9] 졸업논문/model/oversampling/B-SMOTE/SAE/model_SAE_bsmote_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model = load_model(model_name)

        print(isub)
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

        train_vali_data = np.concatenate((tar_data, nontar_data))
        train_vali_label = np.concatenate((tar_label, nontar_label))

        ori_shape = train_vali_data.shape
        reshape_data = np.reshape(train_vali_data, (train_vali_data.shape[0], (train_vali_data.shape[1] * train_vali_data.shape[2])))
        data_res, y_res = adasyn.fit_resample(reshape_data, train_vali_label)
        data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))
        train_data, vali_data, train_label, vali_label = train_test_split(data_res, y_res, test_size=0.10, random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

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
        filename = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/' \
                    'SAE_bsmote_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model

    for isub in range(14):
        adasyn = BorderlineSMOTE(random_state=5)
        model_name = 'E:/[9] 졸업논문/model/oversampling/B-SMOTE/SAE/model_BS_SAE_bsmote_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model = load_model(model_name)

        print(isub)
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

        ori_shape = train_vali_data.shape
        reshape_data = np.reshape(train_vali_data, (train_vali_data.shape[0], (train_vali_data.shape[1] * train_vali_data.shape[2])))
        data_res, y_res = adasyn.fit_resample(reshape_data, train_vali_label)
        data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))

        train_data, vali_data, train_label, vali_label = train_test_split(data_res, y_res, test_size=0.10, random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        total_label = list()
        total_class = list()
        for i in range(ntest):
            test = data2['ERP'][:,150:,:,i]
            total_prob = list()
            for j in range(6):
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
        filename = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/' \
                    'SAE_BS_bsmote_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model