from scipy import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

path = 'C:/Users/jhpark/Desktop/me1/theta.mat'
data = io.loadmat(path)
label = pd.read_csv('C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/label.txt',header=None, engine='python')

from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

re_data = np.reshape(data['theta'][:,150:,:],(500,1000,29))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

label = label.values
for train, test in kfold.split(re_data, label):
    ## standardScaler 해줘보자
    scalers = {}
    for i in range(re_data[train].shape[1]):
        scalers[i] = StandardScaler()
        re_data[train][:, i, :] = scalers[i].fit_transform(re_data[train][:, i, :])

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=20, input_shape=(1000, 29)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=64, kernel_size=20))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(re_data[train], label[train], epochs=50, batch_size=20)
    # evaluate the model

    for k in range(re_data[test].shape[1]):
        re_data[test][:, k, :] = scalers[k].transform(re_data[test][:, k, :])

    scores = model.evaluate(re_data[test], label[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

mean_score = np.mean(cvscores)
print(cvscores)
print(mean_score)