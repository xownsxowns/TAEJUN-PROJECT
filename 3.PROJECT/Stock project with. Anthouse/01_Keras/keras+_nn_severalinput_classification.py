import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv,Series, DataFrame
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas.io.data as web
import datetime
import tensorflow as tf
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
import math
from keras.layers.normalization import BatchNormalization
np.set_printoptions(threshold=np.inf)

class KOSPIDATA:

    def __init__(self):
        start = datetime.datetime(1998, 5, 1)
        end = datetime.datetime(2016, 12, 31)
        kospi = web.DataReader("^KS11", "yahoo", start, end)

        self.arr_date= np.array(kospi.index)
        self.arr_open = np.array(kospi['Open'], dtype=float)
        self.arr_close= np.array(kospi['Adj Close'], dtype=float)
        self.arr_high= np.array(kospi.High, dtype=float)
        self.arr_low= np.array(kospi.Low, dtype=float)
        self.arr_volume= np.array(kospi.Volume, dtype=float)


def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)

        if dataset[i + look_back] > dataset[i + look_back - 1]:
            dataY.append(1)
        elif dataset[i + look_back] <= dataset[i + look_back - 1]:
            dataY.append(0)

    return np.array(dataX), np.array(dataY)

K = KOSPIDATA()

# FEATURES = ['high', 'low', 'open', 'close', 'volume']
FEATURES = ['high', 'low', 'open', 'close']

data = {'year': K.arr_date,
        'open': K.arr_open,
        'high': K.arr_high,
        'low': K.arr_low,
        'close': K.arr_close}

# df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close'])
df = DataFrame(data, columns=['year','open','high','low','close'])

dataset = df['close'].values
print(dataset)

np.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = df[0:train_size], df[train_size:len(dataset)]

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(trainX.shape, trainY.shape)
print(trainX)

model = Sequential()
model.add(Dense(10, input_dim=look_back, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.05)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
history = model.fit(trainX, trainY, validation_split=0.33, nb_epoch=5, batch_size=10, verbose=0)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))