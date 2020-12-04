
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
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.optimizers import Adam
from imblearn.under_sampling import *
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
import keras.backend as K

# parameter setting

np.random.seed(0)
random.seed(0)

for repeat_num in range(1,6):
    total_acc = list()
    train_score = list()
    for isub in range(30,60):
        sm = NeighbourhoodCleaningRule(random_state=5)
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
        data_res, y_res = sm.fit_resample(reshape_data, train_vali_label)
        data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))
        train_data, vali_data, train_label, vali_label = train_test_split(data_res, y_res, test_size=0.10, random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
        vali_data = np.reshape(vali_data, (vali_data.shape[0], vali_data.shape[1] * vali_data.shape[2]))

        input_img = Input(shape=(train_data.shape[1],))

        encoded = Dense(units=(int(train_data.shape[1] / 2)), activation='tanh')(input_img)
        encoded = Dense(units=(int(train_data.shape[1] / 4)), activation='tanh')(encoded)
        encoded = Dense(units=(int(train_data.shape[1] / 8)), activation='tanh')(encoded)
        decoded = Dense(units=(int(train_data.shape[1] / 4)), activation='tanh')(encoded)
        decoded = Dense(units=(int(train_data.shape[1] / 2)), activation='tanh')(decoded)
        decoded = Dense(units=train_data.shape[1], activation='tanh')(decoded)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        # sgd1 = SGD(lr=.1, decay=0.001, momentum=0.9, nesterov=True)
        autoencoder.summary()
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        early_stopping = EarlyStopping(patience=5)
        autoencoder.fit(train_data, train_data, epochs=200, batch_size=8, shuffle=True,
                        validation_data=(vali_data, vali_data), callbacks=[early_stopping])

        for layer in autoencoder.layers[:-3]:
            layer.trainable = False

        new_input = autoencoder.input
        dense1 = Dense(100, activation='relu')(encoded)
        dense2 = Dense(50, activation='relu')(dense1)
        new_output = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))(dense2)

        model = Model(new_input, new_output)
        model.summary()

        model.layers[1].set_weights(autoencoder.layers[1].get_weights())
        model.layers[2].set_weights(autoencoder.layers[2].get_weights())
        model.layers[3].set_weights(autoencoder.layers[3].get_weights())

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(loss='hinge', optimizer=adam, metrics=['accuracy'])
        early_stopping2 = EarlyStopping(patience=10)
        model.fit(train_data, train_label, epochs=200, batch_size=8, validation_data=(vali_data, vali_label),
                  callbacks=[early_stopping2])

        model_name = '//192.168.1.181/office/[04] 연구원개인문서함/이태준/[9] 졸업논문/model/undersampling/NCR/SAE/model_SAE_ncr_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        prob_predicted = model.predict(train_data)
        prob_predicted_label = list()
        for aaa in range(len(prob_predicted)):
            if prob_predicted[aaa][0] > 0.5:
                prob_predicted_label.append(1)
            else:
                prob_predicted_label.append(0)
        training_score = accuracy_score(train_label, prob_predicted_label)
        train_score.append(training_score)

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
                prob = model.predict(test_data)
                total_prob.append(prob[0][0])

                predicted_class = prob.argmax(axis=-1)
                total_class.append(predicted_class[0])
                if j == (data2['target'][i][0] - 1):
                    total_label.append(1)
                else:
                    total_label.append(0)
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label+1):
                corr_ans += 1

        total_acc.append((corr_ans/ntest)*100)
        print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
        print(total_acc)
        print(np.mean(total_acc))

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/' \
                    'SAE_ncr_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model
        del autoencoder
        del encoder

    for isub in range(14):
        sm = NeighbourhoodCleaningRule(random_state=5)
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
        data_res, y_res = sm.fit_resample(reshape_data, train_vali_label)
        data_res = np.reshape(data_res, (data_res.shape[0], ori_shape[1], ori_shape[2]))

        train_data, vali_data, train_label, vali_label = train_test_split(data_res, y_res, test_size=0.10, random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
        vali_data = np.reshape(vali_data, (vali_data.shape[0], vali_data.shape[1] * vali_data.shape[2]))

        input_img = Input(shape=(train_data.shape[1],))

        encoded = Dense(units=(int(train_data.shape[1] / 2)), activation='tanh')(input_img)
        encoded = Dense(units=(int(train_data.shape[1] / 4)), activation='tanh')(encoded)
        encoded = Dense(units=(int(train_data.shape[1] / 8)), activation='tanh')(encoded)
        decoded = Dense(units=(int(train_data.shape[1] / 4)), activation='tanh')(encoded)
        decoded = Dense(units=(int(train_data.shape[1] / 2)), activation='tanh')(decoded)
        decoded = Dense(units=train_data.shape[1], activation='tanh')(decoded)

        autoencoder2 = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        # sgd1 = SGD(lr=.1, decay=0.001, momentum=0.9, nesterov=True)
        autoencoder2.summary()
        autoencoder2.compile(optimizer='adadelta', loss='mean_squared_error')
        early_stopping = EarlyStopping(patience=5)
        autoencoder2.fit(train_data, train_data, epochs=200, batch_size=8, shuffle=True,
                         validation_data=(vali_data, vali_data), callbacks=[early_stopping])

        for layer in autoencoder2.layers[:-3]:
            layer.trainable = False

        new_input = autoencoder2.input
        dense1 = Dense(100, activation='relu')(encoded)
        dense2 = Dense(50, activation='relu')(dense1)
        new_output = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))(dense2)

        model2 = Model(new_input, new_output)
        model2.summary()

        model2.layers[1].set_weights(autoencoder2.layers[1].get_weights())
        model2.layers[2].set_weights(autoencoder2.layers[2].get_weights())
        model2.layers[3].set_weights(autoencoder2.layers[3].get_weights())

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model2.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
        early_stopping2 = EarlyStopping(patience=10)
        model2.fit(train_data, train_label, epochs=200, batch_size=8, validation_data=(vali_data, vali_label),
                   callbacks=[early_stopping2])

        model_name = '//192.168.1.181/office/[04] 연구원개인문서함/이태준/[9] 졸업논문/model/undersampling/NCR/SAE/model_BS_SAE_ncr_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model2.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        prob_predicted = model2.predict(train_data)
        prob_predicted_label = list()
        for aaa in range(len(prob_predicted)):
            if prob_predicted[aaa][0] > 0.5:
                prob_predicted_label.append(1)
            else:
                prob_predicted_label.append(0)
        training_score = accuracy_score(train_label, prob_predicted_label)
        train_score.append(training_score)

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
                prob = model2.predict(test_data)
                total_prob.append(prob[0][0])

                predicted_class = prob.argmax(axis=-1)
                total_class.append(predicted_class[0])
                if j == (data2['target'][i][0] - 1):
                    total_label.append(1)
                else:
                    total_label.append(0)
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label+1):
                corr_ans += 1

        total_acc.append((corr_ans/ntest)*100)
        print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
        print(total_acc)
        print(np.mean(total_acc))

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/' \
                   'SAE_BS_ncr_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model2
        del autoencoder2
        del encoder

    df = pd.DataFrame(total_acc)
    filename = 'P300_Result_SAE_ncr_t' + str(repeat_num) + '.csv'
    df.to_csv(filename)

    df2 = pd.DataFrame(train_score)
    filename = 'P300_Result_SAE_ncr_t' + str(repeat_num) + '_trainscore.csv'
    df2.to_csv(filename)