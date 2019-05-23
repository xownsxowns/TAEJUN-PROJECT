
## P300 Classification
## SVM vs CNN

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping

total_acc = list()
clf_list = list()
for isub in range(60):
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

    kf = KFold(n_splits=10)

    train_data = np.concatenate((tar_data, nontar_data))
    train_label = np.concatenate((tar_label, nontar_label))

    svm_total = list()
    cnn_total = list()

    for train, test in kf.split(train_data):

        X_train, X_test, y_train, y_test = train_data[train], train_data[test], train_label[train], train_label[test]

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(X_train.shape[1]):
            scalers[i] = StandardScaler()
            X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
            X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

        svm_train_data = X_train.reshape((X_train.shape[0], (X_train.shape[1] * X_train.shape[2])))
        svm_test_data = X_test.reshape((X_test.shape[0], (X_test.shape[1] * X_test.shape[2])))
        clf = SVC(probability=True)
        clf.fit(svm_train_data, y_train)
        svm_score = clf.score(svm_test_data, y_test)
        svm_total.append(svm_score)
        print('svm is over: {0}'.format(isub))

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=10 , input_shape=(nlen, nch)))
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
        model.fit(X_train, y_train, epochs=200, batch_size=20, validation_data=(X_test, y_test), callbacks=[early_stopping])
        _, cnn_acc = model.evaluate(X_test, y_test, verbose=0)
        cnn_total.append(cnn_acc)
        print('cnn is over: {0}'.format(isub))

    if np.mean(svm_total) >= np.mean(cnn_total):
        ## train
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

        train_data = np.concatenate((tar_data, nontar_data))
        train_label = np.concatenate((tar_label, nontar_label))

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])

        new_train_data = train_data.reshape((train_data.shape[0], (train_data.shape[1] * train_data.shape[2])))
        clf = SVC(probability=True)
        clf.fit(new_train_data, train_label)

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
                    test_data[:, k, :] = scalers[i].transform(test_data[:, k, :])
                new_test_data = test_data.reshape((test_data.shape[0], (test_data.shape[1] * test_data.shape[2])))
                prob = clf.predict_proba(new_test_data)
                # prob = clf.predict(new_test_data)
                total_prob.append(prob[0][0])
            predicted_label = np.argmin(total_prob)
            if data2['target'][i][0] == (predicted_label+1):
                corr_ans += 1

        total_acc.append((corr_ans/ntest)*100)
        clf_list.append('svm')
        print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
        print(total_acc)
    elif np.mean(svm_total) < np.mean(cnn_total):
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
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

        train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label,
                                                                          test_size=0.15, random_state=42)

        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=10, input_shape=(nlen, nch)))
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
        model.fit(train_data, train_label, epochs=200, batch_size=20, validation_data=(vali_data, vali_label),
                  callbacks=[early_stopping])

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub + 1) + '_EP_test.mat'
        # path = '/Volumes/TAEJUN/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
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
                    test_data[:, k, :] = scalers[i].transform(test_data[:, k, :])
                prob = model.predict_proba(test_data)
                total_prob.append(prob[0][0])
            predicted_label = np.argmax(total_prob)
            if data2['target'][i][0] == (predicted_label + 1):
                corr_ans += 1

        total_acc.append((corr_ans / ntest) * 100)
        clf_list.append('cnn')
        print("Accuracy: %.2f%%" % ((corr_ans / ntest) * 100))
        print(total_acc)
        print(np.mean(total_acc))

df = pd.DataFrame(total_acc)
dff = pd.DataFrame(clf_list)
filename = 'P300_Result_compare.csv'
filename2 = 'P300_Result_clf.csv'
df.to_csv(filename)
dff.to_csv(filename2)