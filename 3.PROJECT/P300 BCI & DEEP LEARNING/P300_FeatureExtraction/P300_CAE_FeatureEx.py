
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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Input, UpSampling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.optimizers import Adam

np.random.seed(0)
total_acc = list()
ch_kernel_size = (1, 5)
dp_kernel_size = (10, 1)


for isub in range(30,40):
    print(isub+1)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    # path = '/Volumes/TAEJUN_USB/현차_기술과제데이터/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
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

    train_data = np.expand_dims(train_data, axis=1)
    vali_data = np.expand_dims(vali_data, axis=1)

    # ENCODER
    input_img = Input((1, nlen, nch))
    if (nch % 2) == 0:
        x = Conv2D(filters=32, kernel_size=ch_kernel_size, activation='relu', border_mode='same',
                   data_format='channels_first')(input_img)  # nb_filter, nb_row, nb_col
    else:
        x = ZeroPadding2D(padding=((0, 0), (0, 1)), data_format='channels_first')(input_img)
        x = Conv2D(filters=32, kernel_size=ch_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(x)  # nb_filter, nb_row, nb_col
    x = MaxPooling2D(pool_size=(1, 2), border_mode='same', data_format='channels_first')(x)
    x = Conv2D(filters=64, kernel_size=dp_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(x)
    encoded = MaxPooling2D(pool_size=(1, 1), border_mode='same', data_format='channels_first')(x)

    # DECODER
    x = Conv2D(filters=64, kernel_size=dp_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(encoded)
    x = UpSampling2D(size=(1, 2), data_format='channels_first')(x)
    x = Conv2D(filters=32, kernel_size=ch_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(x)
    x = UpSampling2D(size=(1, 1), data_format='channels_first')(x)
    decoded = Conv2D(filters=1, kernel_size=(1,1),  activation='sigmoid', border_mode='same', data_format='channels_first')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(patience=5)
    autoencoder.fit(train_data,train_data, epochs=200, batch_size=8, shuffle=True, validation_data=(vali_data, vali_data), callbacks=[early_stopping])

    for layer in autoencoder.layers[:-5]:
        layer.trainable = False

    new_input = autoencoder.input
    dense1 = Flatten()(encoded)
    # dense2 = Dense(100, activation='tanh')(dense1)
    dense3 = Dense(50, activation='tanh')(dense1)
    new_output = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))(dense3)

    model = Model(new_input, new_output)
    model.summary()

    model.layers[1].set_weights(autoencoder.layers[1].get_weights())
    model.layers[2].set_weights(autoencoder.layers[2].get_weights())
    model.layers[3].set_weights(autoencoder.layers[3].get_weights())
    model.layers[4].set_weights(autoencoder.layers[4].get_weights())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
    model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
    early_stopping2 = EarlyStopping(patience=10)
    model.fit(train_data, train_label, epochs=200, batch_size=8, validation_data=(vali_data, vali_label), callbacks=[early_stopping2])

    model_name = 'E:/[9] 졸업논문/model/model_CAE_train'+str(isub+1)+'.h5'
    model.save(model_name)

    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    # path = '/Volumes/UNTITLED2/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
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
            prob = model.predict(test_data)
            total_prob.append(prob[0][0])
        predicted_label = np.argmax(total_prob)
        if data2['target'][i][0] == (predicted_label+1):
            corr_ans += 1

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))
#
# for isub in range(14):
#     print(isub)
#     path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
#     # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
#     data = io.loadmat(path)
#
#     nch = np.shape(data['ERP'])[0]
#     nlen = 250
#     ntrain = np.shape(data['ERP'])[3]
#
#     tar_data = list()
#     tar_label = list()
#     nontar_data = list()
#     nontar_label = list()
#
#     for i in range(ntrain):
#         target = data['ERP'][:,150:,data['target'][i][0]-1,i]
#         tar_data.append(target)
#         tar_label.append(1)
#
#         for j in range(6):
#             if j == (data['target'][i][0]-1):
#                 continue
#             else:
#                 nontar_data.append(data['ERP'][:,150:,j,i])
#                 nontar_label.append(0)
#
#     tar_data = np.reshape(tar_data,(ntrain,nlen,nch))
#     nontar_data = np.reshape(nontar_data,((ntrain*5),nlen,nch))
#
#     train_vali_data = np.concatenate((tar_data, nontar_data))
#     train_vali_label = np.concatenate((tar_label, nontar_label))
#
#     train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.15, random_state=42)
#
#     ## standardScaler 해줘보자
#     scalers = {}
#     for i in range(train_data.shape[1]):
#         scalers[i] = StandardScaler()
#         train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
#         vali_data[:,i,:] = scalers[i].transform(vali_data[:,i,:])
#
#     train_data = np.expand_dims(train_data, axis=1)
#     vali_data = np.expand_dims(vali_data, axis=1)
#
#     # ENCODER
#     input_img = Input((1, nlen, nch))
#     if (nch % 2) == 0:
#         x = Conv2D(filters=32, kernel_size=ch_kernel_size, activation='relu', border_mode='same',
#                    data_format='channels_first')(input_img)  # nb_filter, nb_row, nb_col
#     else:
#         x = ZeroPadding2D(padding=((0, 0), (0, 1)), data_format='channels_first')(input_img)
#         x = Conv2D(filters=32, kernel_size=ch_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(x)  # nb_filter, nb_row, nb_col
#     x = MaxPooling2D(pool_size=(1, 2), border_mode='same', data_format='channels_first')(x)
#     x = Conv2D(filters=64, kernel_size=dp_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(x)
#     encoded = MaxPooling2D(pool_size=(1, 1), border_mode='same', data_format='channels_first')(x)
#
#     # DECODER
#     x = Conv2D(filters=64, kernel_size=dp_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(encoded)
#     x = UpSampling2D(size=(1, 2), data_format='channels_first')(x)
#     x = Conv2D(filters=32, kernel_size=ch_kernel_size, activation='relu', border_mode='same', data_format='channels_first')(x)
#     x = UpSampling2D(size=(1, 1), data_format='channels_first')(x)
#     decoded = Conv2D(filters=1, kernel_size=(1,1),  activation='sigmoid', border_mode='same', data_format='channels_first')(x)
#
#     autoencoder = Model(input_img, decoded)
#     encoder = Model(input_img, encoded)
#
#     autoencoder.summary()
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#     early_stopping = EarlyStopping(patience=5)
#     autoencoder.fit(train_data,train_data, epochs=200, batch_size=8, shuffle=True, validation_data=(vali_data, vali_data), callbacks=[early_stopping])
#
#     for layer in autoencoder.layers[:-5]:
#         layer.trainable = False
#
#     new_input = autoencoder.input
#     dense1 = Flatten()(encoded)
#     dense2 = Dense(100, activation='relu')(dense1)
#     dense3 = Dense(50, activation='relu')(dense2)
#     new_output = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))(dense3)
#
#     model = Model(new_input, new_output)
#     model.summary()
#
#     model.layers[1].set_weights(autoencoder.layers[1].get_weights())
#     model.layers[2].set_weights(autoencoder.layers[2].get_weights())
#     model.layers[3].set_weights(autoencoder.layers[3].get_weights())
#     model.layers[4].set_weights(autoencoder.layers[4].get_weights())
#
#     adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
#     model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
#     early_stopping2 = EarlyStopping(patience=10)
#     model.fit(train_data, train_label, epochs=200, batch_size=8, validation_data=(vali_data, vali_label), callbacks=[early_stopping2])
#
#     model_name = 'E:/[9] 졸업논문/model/model_BS_CAE_train'+str(isub+1)+'.h5'
#     model.save(model_name)
#
#     ## Test
#     path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
#     # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
#     data2 = io.loadmat(path)
#     corr_ans = 0
#     ntest = np.shape(data2['ERP'])[3]
#
#     for i in range(ntest):
#         test = data2['ERP'][:,150:,:,i]
#         total_prob = list()
#         for j in range(6):
#             test_data = test[:,:,j]
#             test_data = np.reshape(test_data, (1,nlen,nch))
#             for k in range(test_data.shape[1]):
#                 test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
#             test_data = np.expand_dims(test_data, axis=1)
#             prob = model.predict(test_data)
#             total_prob.append(prob[0][0])
#         predicted_label = np.argmax(total_prob)
#         if data2['target'][i][0] == (predicted_label+1):
#             corr_ans += 1
#
#     total_acc.append((corr_ans/ntest)*100)
#     print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
#     print(total_acc)
#     print(np.mean(total_acc))

df = pd.DataFrame(total_acc)
filename = 'P300_Result_CAE1.csv'
df.to_csv(filename)