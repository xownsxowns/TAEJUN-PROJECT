
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
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import keras
import gc
import keras.backend as K

np.random.seed(0)
total_acc = list()

for isub in range(30,60):
    print(isub)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
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

    train_vali_data = np.concatenate((tar_data, nontar_data))
    train_vali_label = np.concatenate((tar_label, nontar_label))

    train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.10, random_state=42)

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
        vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])


    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    vali_data = np.reshape(vali_data, (vali_data.shape[0], vali_data.shape[1]*vali_data.shape[2]))

    input_img = Input(shape=(train_data.shape[1],))

    encoded = Dense(units=3000, activation='tanh')(input_img)
    # encoded = Dense(units=3000, activation='relu')(encoded)
    encoded = Dense(units=500, activation='tanh')(encoded)
    # encoded = Dense(units=500, activation='relu')(encoded)
    # decoded = Dense(units=1000, activation='relu')(encoded)
    # decoded = Dense(units=3000, activation='relu')(decoded)
    decoded = Dense(units=3000, activation='tanh')(encoded)
    decoded = Dense(units=train_data.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(train_data,train_data, epochs=200, batch_size=8, shuffle=True, validation_data=(vali_data, vali_data))

    for layer in autoencoder.layers[:-3]:
        layer.trainable = False

    autoencoder.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

    new_input = autoencoder.input
    hidden_layer = autoencoder.layers[-3].output

    del autoencoder
    gc.collect()

    dense1 = Dense(300, activation='relu')(hidden_layer)
    dense2 = Dense(100, activation='relu')(dense1)
    new_output = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))(dense2)
    model = Model(new_input, new_output)
    model.summary()
    model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=10)
    model.fit(train_data, train_label, epochs=200, batch_size=8, validation_data=(vali_data, vali_label), callbacks=[early_stopping])

    model_name = 'model_SAE_train'+str(isub+1)+'.h5'
    model.save(model_name)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
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
                test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
            prob = model.predict(test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))
    K.clear_session()


for isub in range(14):
    print(isub)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
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

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    vali_data = np.reshape(vali_data, (vali_data.shape[0], vali_data.shape[1]*vali_data.shape[2]))

    input_img = Input(shape=(train_data.shape[1],))

    encoded = Dense(units=3000, activation='tanh')(input_img)
    # encoded = Dense(units=3000, activation='relu')(encoded)
    encoded = Dense(units=500, activation='tanh')(encoded)
    # encoded = Dense(units=500, activation='relu')(encoded)
    # decoded = Dense(units=1000, activation='relu')(encoded)
    # decoded = Dense(units=3000, activation='relu')(decoded)
    decoded = Dense(units=3000, activation='tanh')(encoded)
    decoded = Dense(units=train_data.shape[1], activation='sigmoid')(decoded)

    autoencoder2 = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder2.summary()
    autoencoder2.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder2.fit(train_data,train_data, epochs=200, batch_size=8, shuffle=True, validation_data=(vali_data, vali_data))

    for layer in autoencoder2.layers[:-3]:
        layer.trainable = False

    autoencoder2.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

    new_input = autoencoder2.input
    hidden_layer = autoencoder2.layers[-3].output

    del autoencoder2
    gc.collect()

    dense1 = Dense(300, activation='relu')(hidden_layer)
    dense2 = Dense(100, activation='relu')(dense1)
    new_output = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))(dense2)
    model = Model(new_input, new_output)
    model.summary()
    model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=10)
    model.fit(train_data, train_label, epochs=200, batch_size=8, validation_data=(vali_data, vali_label), callbacks=[early_stopping])

    model_name = 'model_BS_SAE_train'+str(isub+1)+'.h5'
    model.save(model_name)
    ## classifier

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
    # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
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
            prob = model.predict(test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))
    K.clear_session()

df = pd.DataFrame(total_acc)
filename = 'P300_Result2_SAE.csv'
df.to_csv(filename)