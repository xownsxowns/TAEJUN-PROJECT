
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


from scipy import io
import pandas as pd
import numpy as np
import PIL.Image as pilimg
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField

import tensorflow as tf
from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


total_acc = list()

for isub in range(60):
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

    # 100ms~600ms 길이 자른것
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

    nontar_trial = list()
    # non target data
    for itrial in range(nontar_data.shape[0]):
        for ich in range(nontar_data.shape[2]):
            file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub + 1) + '/nontar_trial' + str(itrial + 1) + '_GAF_ch' + str(ich + 1) + '.png'
            if ich == 0:
                image_open = pilimg.open(file_path)
                image_open = image_open.resize((64,64))
                image = np.array(image_open)[...,:3]
                image_total = image
            else:
                image_open = pilimg.open(file_path)
                image_open = image_open.resize((64, 64))
                image = np.array(image_open)[...,:3]
                image_total = np.concatenate((image_total, image), axis=2)
        nontar_trial.append(image_total)
        print('{0} nontar trial ended'.format(itrial+1))

    tar_trial = list()
    # target data
    for itrial in range(tar_data.shape[0]):
        for ich in range(tar_data.shape[2]):
            file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub + 1) + '/tar_trial' + str(itrial + 1) + '_GAF_ch' + str(ich + 1) + '.png'
            if ich == 0:
                image_open = pilimg.open(file_path)
                image_open = image_open.resize((64,64))
                image = np.array(image_open)[...,:3]
                image_total = image
            else:
                image_open = pilimg.open(file_path)
                image_open = image_open.resize((64, 64))
                image = np.array(image_open)[...,:3]
                image_total = np.concatenate((image_total, image), axis=2)
        tar_trial.append(image_total)
        print('{0} tar trial ended'.format(itrial+1))

    train_vali_data = np.concatenate((tar_trial, nontar_trial))
    train_vali_label = np.concatenate((tar_label, nontar_label))

    train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.15, random_state=42)
    nch = tar_data.shape[2]

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, (nch*3)), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    # early_stopping = EarlyStopping(patience=20)
    # model.fit(train_data, train_label, epochs=500, batch_size=20, validation_data=(vali_data, vali_label), callbacks=[early_stopping])
    model.fit(train_data, train_label, epochs=500, batch_size=20)
    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntest = np.shape(data['ERP'])[3]
    nstim = 4

    test_data = list()
    # ntest, nch, nlen, nstim
    # 100ms~600ms 길이 자른것
    for i in range(ntest):
        target = data['ERP'][:, 150:, :, i]
        test_data.append(target)
    # nstim, ntest, nlen, nch
    test_data = np.transpose(test_data, (3, 0, 2, 1))

    corr_ans = 0
    for itrial in range(test_data.shape[1]):
        stim1_trial = list()
        stim2_trial = list()
        stim3_trial = list()
        stim4_trial = list()
        total_prob = list()
        for nstim in range(test_data.shape[0]):
            for ich in range(tar_data.shape[2]):
                file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub + 1) + '/test_trial' + str(nstim+1) + '-' + str(itrial+1) + '_GAF_ch' + str(ich+1) + '.png'
                if ich == 0:
                    image_open = pilimg.open(file_path)
                    image_open = image_open.resize((64,64))
                    image = np.array(image_open)[...,:3]
                    image_total = image
                else:
                    image_open = pilimg.open(file_path)
                    image_open = image_open.resize((64, 64))
                    image = np.array(image_open)[...,:3]
                    image_total = np.concatenate((image_total, image), axis=2)
            if nstim == 0:
                stim1_trial.append(image_total)
            elif nstim == 1:
                stim2_trial.append(image_total)
            elif nstim == 2:
                stim3_trial.append(image_total)
            elif nstim == 3:
                stim4_trial.append(image_total)

        total_trial = np.concatenate((stim1_trial,stim2_trial,stim3_trial,stim4_trial))
        prob = model.predict_proba(total_trial)
        total_prob.append(prob[0][0])
        total_prob.append(prob[1][0])
        total_prob.append(prob[2][0])
        total_prob.append(prob[3][0])

        predicted_label = np.argmax(total_prob)
        if data['target'][itrial][0] == (predicted_label+1):
            corr_ans += 1

        print('sub{0}: {1} test trial ended'.format(isub+1, itrial+1))

    total_acc.append((corr_ans/ntest)*100)
    print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
    print(total_acc)
    print(np.mean(total_acc))

df = pd.DataFrame(total_acc)
filename = 'P300_Result_CNN_GAF.csv'
df.to_csv(filename)