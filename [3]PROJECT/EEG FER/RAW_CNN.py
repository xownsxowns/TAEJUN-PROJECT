import numpy as np
import _pickle as cPickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv

# data: 40x40x8064 [video/trial x channel(~32:EEG) x data]
# channel: Geneva format
# labels: 40x4 [video/trial x label(valence,arousal,dominance,liking)]
# Downsampled: 128Hz, Bandpass: 4.0~45.0Hz, Data length: 63s (3 second pre-trial baseline)
# filepath = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s01.dat'
# x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
# data = x['data']
# label = x['labels']

# path
lab_path = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s'
mac_path = '/Volumes/TAEJUN_USB/DEAP/data_preprocessed_python/data_preprocessed_python/s'

# label & data load
path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/EEG FER/'
read_emotion_label = np.load(path + 'emotion_label.npy', allow_pickle=True).item()

acc = list()
predict = np.zeros((22,40))
passsub = [3,5,11,14]

for itrial in range(1,23):
    if itrial in passsub:
        continue
    test_data = np.array([])
    train_data = np.array([])
    for isub in range(1,23):
        if isub == itrial:
            filepath = lab_path + str(isub).zfill(2) + '.dat'
            x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
            test_data = x['data']
            test_label = read_emotion_label[isub]
            print(isub)
        elif isub not in passsub:
            filepath = lab_path + str(isub).zfill(2) + '.dat'
            x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
            if train_data.size == 0:
                train_data = x['data']
                label = read_emotion_label[isub]
            else:
                train_data = np.concatenate((train_data, x['data']))
                label = np.concatenate((label, read_emotion_label[isub]))
            print(isub)

    ## feature extraction
    # remove baseline, except EEG signal
    # channel: 32, length: 7680
    train_data = train_data[:,:32,384:]
    test_data  = test_data[:,:32,384:]

    train_data = np.transpose(train_data, (0,2,1))
    test_data  = np.transpose(test_data, (0,2,1))

    train, vali, train_label, vali_label = train_test_split(train_data, label, test_size=0.15, random_state=42)

    ## standardScaler 해줘보자
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train[:, i, :] = scalers[i].fit_transform(train[:, i, :])
        vali[:,i,:] = scalers[i].transform(vali[:,i,:])

    nlen = np.shape(train)[1]
    nch  = np.shape(train)[2]

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
    model.add(Dense(7, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    early_stopping = EarlyStopping(patience=10)
    model.fit(train, train_label, epochs=200, batch_size=10, validation_data=(vali, vali_label), callbacks=[early_stopping])
    # overfitting 되는 것 같음

    correct_ans = 0
    for i in range(test_data.shape[0]):
        for k in range(test_data.shape[1]):
            test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])
        predicted_label = model.predict_classes(np.expand_dims(test_data[i,:,:],axis=0))[0]
        if predicted_label == test_label[i]:
            correct_ans += 1
        predict[itrial-1, i] = predicted_label

    acc.append(correct_ans/len(test_data))
    print(acc)

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/EEG FER/'

f = open(path+'ACC_CNN.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(acc)
f.close()

np.savetxt(path+'predicted_label_cnn.csv',predict,fmt="%d",delimiter=",")



