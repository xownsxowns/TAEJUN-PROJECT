
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
import gc
import keras.backend as K

# parameter setting

train_num = [10, 20, 30, 40, 50]
for aaa in range(5):

    total_acc = list()
    train_score = list()
    train_score_prob = list()
    np.random.seed(0)
    random.seed(0)

    ntrain = train_num[aaa]
    for isub in range(30,60):
        print(isub)
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
        data = io.loadmat(path)

        nch = np.shape(data['ERP'])[0]
        nlen = 250

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

        train_data = np.expand_dims(train_data, axis=1)
        vali_data = np.expand_dims(vali_data, axis=1)

        ch_kernel_size = (1, nch)
        dp_kernel_size = (10, 1)

        ## Build Stacked AutoEncoder
        model = Sequential()
        # channel convolution
        model.add(Conv2D(filters=32, kernel_size=ch_kernel_size, input_shape=(1, nlen, nch), data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(1,2), data_format='channels_first'))
        # data point convolution
        model.add(Conv2D(filters=64, kernel_size=dp_kernel_size, data_format='channels_first', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(1,1), data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        early_stopping = EarlyStopping(patience=5)
        model.fit(train_data, train_label, epochs=200, batch_size=30, validation_data=(vali_data, vali_label), callbacks=[early_stopping])

        model_name = 'E:/[9] 졸업논문/model/train_num/model_CNN2_' + str(ntrain) + '_train' + str(isub + 1) + '.h5'
        model.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        training_score = accuracy_score(train_label, model.predict_classes(train_data))
        train_score.append(training_score)

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
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
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
                test_data = np.expand_dims(test_data, axis=1)
                prob = model.predict_proba(test_data)
                total_prob.append(prob[0][0])
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label+1):
                corr_ans += 1

        total_acc.append((corr_ans/ntest)*100)
        print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
        print(total_acc)
        print(np.mean(total_acc))

    for isub in range(14):
        print(isub)
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
        data = io.loadmat(path)

        nch = np.shape(data['ERP'])[0]
        nlen = 250

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

        train_data = np.expand_dims(train_data, axis=1)
        vali_data = np.expand_dims(vali_data, axis=1)

        ch_kernel_size = (1, nch)
        dp_kernel_size = (10, 1)
        ## Build Stacked AutoEncoder
        model = Sequential()
        # channel convolution
        model.add(Conv2D(filters=32, kernel_size=ch_kernel_size, input_shape=(1, nlen, nch), data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(1, 2), data_format='channels_first'))
        # data point convolution
        model.add(Conv2D(filters=64, kernel_size=dp_kernel_size, data_format='channels_first', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(1,1), data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        early_stopping = EarlyStopping(patience=5)
        model.fit(train_data, train_label, epochs=200, batch_size=30, validation_data=(vali_data, vali_label), callbacks=[early_stopping])

        model_name = 'E:/[9] 졸업논문/model/train_num/model_BS_CNN2_' + str(ntrain) + '_train' + str(isub + 1) + '.h5'
        model.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        training_score = accuracy_score(train_label, model.predict_classes(train_data))
        train_score.append(training_score)

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
                test_data = np.expand_dims(test_data, axis=1)
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
    filename = 'P300_Result_CNN2_' + str(ntrain) + '.csv'
    df.to_csv(filename)

    df2 = pd.DataFrame(train_score)
    filename = 'P300_Result_CNN2_' + str(ntrain) + '_trainscore.csv'
    df2.to_csv(filename)

    K.clear_session()
    gc.collect()
    del model