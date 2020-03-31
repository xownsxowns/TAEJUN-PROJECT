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
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, LSTM, \
    Input, Permute
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import *
import gc
import keras.backend as K

# parameter setting

np.random.seed(0)
random.seed(0)

for repeat_num in range(1, 11):
    total_acc = list()
    train_score = list()
    for isub in range(30, 60):
        sm = TomekLinks(random_state=5)
        print(isub)
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
        data = io.loadmat(path)

        nch = np.shape(data['ERP'])[0]
        nlen = 250
        ntrain = np.shape(data['ERP'])[3]

        tar_data = list()
        tar_label = list()
        nontar_data = list()
        nontar_label = list()

        for i in range(ntrain):
            target = data['ERP'][:, 150:, data['target'][i][0] - 1, i]
            tar_data.append(target)
            tar_label.append(1)

            for j in range(4):
                if j == (data['target'][i][0] - 1):
                    continue
                else:
                    nontar_data.append(data['ERP'][:, 150:, j, i])
                    nontar_label.append(0)

        tar_data = np.reshape(tar_data, (ntrain, nlen, nch))
        nontar_data = np.reshape(nontar_data, ((ntrain * 3), nlen, nch))

        train_vali_data = np.concatenate((tar_data, nontar_data))
        train_vali_label = np.concatenate((tar_label, nontar_label))

        ori_shape = train_vali_data.shape
        reshape_data = np.reshape(train_vali_data,
                                  (train_vali_data.shape[0], (train_vali_data.shape[1] * train_vali_data.shape[2])))
        data_res, y_res = sm.fit_resample(reshape_data, train_vali_label)
        data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))
        train_data, vali_data, train_label, vali_label = train_test_split(data_res, y_res, test_size=0.10,
                                                                          random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        train_data = np.expand_dims(train_data, axis=1)
        vali_data = np.expand_dims(vali_data, axis=1)

        ch_kernel_size = (1, nch)
        dp_kernel_size = (10, 1)

        input_img = Input(shape=(1, nlen, nch))
        x = Conv2D(filters=32, kernel_size=ch_kernel_size, data_format='channels_first')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Permute((2, 3, 1))(x)
        x = Reshape((nlen, 32))(x)
        ## Build Stacked AutoEncoder

        # LSTM
        x = LSTM(64)(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

        LSTM_CNN = Model(input_img, x)
        LSTM_CNN.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
        print(LSTM_CNN.summary())
        early_stopping = EarlyStopping(patience=5)
        LSTM_CNN.fit(train_data, train_label, epochs=200, batch_size=30, validation_data=(vali_data, vali_label),
                     callbacks=[early_stopping])

        model_name = 'E:/[9] 졸업논문/model/undersampling/Tomek/ConvLSTM/model_CNN_LSTM_tomek_t' + str(repeat_num) + '_train' + str(
            isub + 1) + '.h5'
        LSTM_CNN.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        prob_predicted = LSTM_CNN.predict(train_data)
        prob_predicted_label = list()
        for aaa in range(len(prob_predicted)):
            if prob_predicted[aaa][0] > 0.5:
                prob_predicted_label.append(1)
            else:
                prob_predicted_label.append(0)
        training_score = accuracy_score(train_label, prob_predicted_label)
        train_score.append(training_score)

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub + 1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        for i in range(ntest):
            test = data2['ERP'][:, 150:, :, i]
            total_prob = list()
            for j in range(4):
                test_data = test[:, :, j]
                test_data = np.reshape(test_data, (1, nlen, nch))
                for k in range(test_data.shape[1]):
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
                test_data = np.expand_dims(test_data, axis=1)
                prob = LSTM_CNN.predict(test_data)
                total_prob.append(prob[0][0])
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label + 1):
                corr_ans += 1

        total_acc.append((corr_ans / ntest) * 100)
        print("Accuracy: %.2f%%" % ((corr_ans / ntest) * 100))
        print(total_acc)
        print(np.mean(total_acc))

        K.clear_session()
        gc.collect()
        del LSTM_CNN

    for isub in range(14):
        sm = TomekLinks(random_state=5)
        print(isub)
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub + 1) + '_EP_training.mat'
        data = io.loadmat(path)

        nch = np.shape(data['ERP'])[0]
        nlen = 250
        ntrain = np.shape(data['ERP'])[3]

        tar_data = list()
        tar_label = list()
        nontar_data = list()
        nontar_label = list()

        for i in range(ntrain):
            target = data['ERP'][:, 150:, data['target'][i][0] - 1, i]
            tar_data.append(target)
            tar_label.append(1)

            for j in range(6):
                if j == (data['target'][i][0] - 1):
                    continue
                else:
                    nontar_data.append(data['ERP'][:, 150:, j, i])
                    nontar_label.append(0)

        tar_data = np.reshape(tar_data, (ntrain, nlen, nch))
        nontar_data = np.reshape(nontar_data, ((ntrain * 5), nlen, nch))

        train_vali_data = np.concatenate((tar_data, nontar_data))
        train_vali_label = np.concatenate((tar_label, nontar_label))

        ori_shape = train_vali_data.shape
        reshape_data = np.reshape(train_vali_data,
                                  (train_vali_data.shape[0], (train_vali_data.shape[1] * train_vali_data.shape[2])))
        data_res, y_res = sm.fit_resample(reshape_data, train_vali_label)
        data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))

        train_data, vali_data, train_label, vali_label = train_test_split(data_res, y_res, test_size=0.10,
                                                                          random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        train_data = np.expand_dims(train_data, axis=1)
        vali_data = np.expand_dims(vali_data, axis=1)

        ch_kernel_size = (1, nch)
        dp_kernel_size = (10, 1)

        input_img = Input(shape=(1, nlen, nch))
        x = Conv2D(filters=32, kernel_size=ch_kernel_size, data_format='channels_first')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Permute((2, 3, 1))(x)
        x = Reshape((nlen, 32))(x)
        ## Build Stacked AutoEncoder

        # LSTM
        x = LSTM(64)(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

        LSTM_CNN = Model(input_img, x)
        LSTM_CNN.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
        print(LSTM_CNN.summary())
        early_stopping = EarlyStopping(patience=5)
        LSTM_CNN.fit(train_data, train_label, epochs=200, batch_size=30, validation_data=(vali_data, vali_label),
                     callbacks=[early_stopping])

        model_name = 'E:/[9] 졸업논문/model/undersampling/Tomek/ConvLSTM/model_BS_CNN_LSTM_tomek_t' + str(
            repeat_num) + '_train' + str(isub + 1) + '.h5'
        LSTM_CNN.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        prob_predicted = LSTM_CNN.predict(train_data)
        prob_predicted_label = list()
        for aaa in range(len(prob_predicted)):
            if prob_predicted[aaa][0] > 0.5:
                prob_predicted_label.append(1)
            else:
                prob_predicted_label.append(0)
        training_score = accuracy_score(train_label, prob_predicted_label)
        train_score.append(training_score)

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub + 1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        for i in range(ntest):
            test = data2['ERP'][:, 150:, :, i]
            total_prob = list()
            for j in range(6):
                test_data = test[:, :, j]
                test_data = np.reshape(test_data, (1, nlen, nch))
                for k in range(test_data.shape[1]):
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
                test_data = np.expand_dims(test_data, axis=1)
                prob = LSTM_CNN.predict(test_data)
                total_prob.append(prob[0][0])
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label + 1):
                corr_ans += 1

        total_acc.append((corr_ans / ntest) * 100)
        print("Accuracy: %.2f%%" % ((corr_ans / ntest) * 100))
        print(total_acc)
        print(np.mean(total_acc))

        K.clear_session()
        gc.collect()
        del LSTM_CNN

    df = pd.DataFrame(total_acc)
    filename = 'P300_Result_CNN_LSTM_tomek_t' + str(repeat_num) + '.csv'
    df.to_csv(filename)

    df2 = pd.DataFrame(train_score)
    filename = 'P300_Result_CNN_LSTM_tomek_t' + str(repeat_num) + '_trainscore.csv'
    df2.to_csv(filename)