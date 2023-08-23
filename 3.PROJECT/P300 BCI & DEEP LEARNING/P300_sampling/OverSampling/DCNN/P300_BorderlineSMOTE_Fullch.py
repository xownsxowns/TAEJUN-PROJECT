
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
from keras import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Add, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
import keras.backend as K
from imblearn.over_sampling import *

# parameter setting

np.random.seed(0)
random.seed(0)

# ResNet
def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x

def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)
    shortcut = x
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if (i == 0):
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if (i == 0):
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

for repeat_num in range(1,6):
    total_acc = list()
    train_score = list()
    for isub in range(30,60):
        sm = BorderlineSMOTE(random_state=5)
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
            vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])

        train_data = np.expand_dims(train_data, axis=3)
        vali_data = np.expand_dims(vali_data, axis=3)

        input_tensor = Input(shape=(250, nch, 1), dtype='float32', name='input')
        class_num = 1

        x = conv1_layer(input_tensor)
        x = conv2_layer(x)
        x = conv3_layer(x)
        x = conv4_layer(x)
        x = conv5_layer(x)

        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(class_num, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

        resnet50 = Model(input_tensor, output_tensor)
        resnet50.summary()
        resnet50.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(patience=5)
        resnet50.fit(train_data, train_label, epochs=200, batch_size=30, validation_data=(vali_data, vali_label),
                     callbacks=[early_stopping])

        model_name = '//192.168.1.181/office/[04] 연구원개인문서함/이태준/[9] 졸업논문/model/oversampling/B-SMOTE/DCNN/model_DCNN_bsmote_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        resnet50.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        prob_predicted = resnet50.predict(train_data)
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
            test = data2['ERP'][:, 150:, :, i]
            total_prob = list()
            for j in range(4):
                test_data = test[:, :, j]
                test_data = np.reshape(test_data, (1, nlen, nch))
                for k in range(test_data.shape[1]):
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
                test_data = np.expand_dims(test_data, axis=3)
                prob = resnet50.predict(test_data)
                total_prob.append(prob[0][0])

                predicted_class = prob.argmax(axis=-1)
                total_class.append(predicted_class[0])
                if j == (data2['target'][i][0]-1):
                    total_label.append(1)
                else:
                    total_label.append(0)
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label + 1):
                corr_ans += 1

        total_acc.append((corr_ans/ntest)*100)
        print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
        print(total_acc)
        print(np.mean(total_acc))

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/' \
                    'DCNN_bsmote_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del resnet50

    for isub in range(14):
        sm = BorderlineSMOTE(random_state=5)
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
            vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])

        train_data = np.expand_dims(train_data, axis=3)
        vali_data = np.expand_dims(vali_data, axis=3)

        input_tensor = Input(shape=(250, nch, 1), dtype='float32', name='input')
        class_num = 1

        x = conv1_layer(input_tensor)
        x = conv2_layer(x)
        x = conv3_layer(x)
        x = conv4_layer(x)
        x = conv5_layer(x)

        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(class_num, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

        resnet50 = Model(input_tensor, output_tensor)
        resnet50.summary()
        resnet50.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(patience=5)
        resnet50.fit(train_data, train_label, epochs=200, batch_size=30, validation_data=(vali_data, vali_label),
                     callbacks=[early_stopping])

        model_name = '//192.168.1.181/office/[04] 연구원개인문서함/이태준/[9] 졸업논문/model/oversampling/B-SMOTE/DCNN/model_BS_DCNN_bsmote_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        resnet50.save(model_name)

        ## prob로 하지 않고 그냥 predict로 했을 때
        prob_predicted = resnet50.predict(train_data)
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
            test = data2['ERP'][:, 150:, :, i]
            total_prob = list()
            for j in range(6):
                test_data = test[:, :, j]
                test_data = np.reshape(test_data, (1, nlen, nch))
                for k in range(test_data.shape[1]):
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
                test_data = np.expand_dims(test_data, axis=3)
                prob = resnet50.predict(test_data)
                total_prob.append(prob[0][0])

                predicted_class = prob.argmax(axis=-1)
                total_class.append(predicted_class[0])
                if j == (data2['target'][i][0]-1):
                    total_label.append(1)
                else:
                    total_label.append(0)
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label + 1):
                corr_ans += 1

        total_acc.append((corr_ans/ntest)*100)
        print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
        print(total_acc)
        print(np.mean(total_acc))

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/OverSampling/CONFUSION/B-SMOTE/' \
                    'DCNN_BS_bsmote_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del resnet50

    df = pd.DataFrame(total_acc)
    filename = 'P300_Result_DCNN_bsmote_t' + str(repeat_num) + '.csv'
    df.to_csv(filename)

    df2 = pd.DataFrame(train_score)
    filename = 'P300_Result_DCNN_bsmote_t' + str(repeat_num) + '_trainscore.csv'
    df2.to_csv(filename)

