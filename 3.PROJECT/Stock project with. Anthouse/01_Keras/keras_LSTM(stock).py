import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import pandas.io.data as web
import datetime
import pandas
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class KOSPIDATA:
    def __init__(self):
        start = datetime.datetime(1998, 5, 1)
        end = datetime.datetime(2016, 12, 31)
        kospi = web.DataReader("KRX:KOSPI", "google", start, end)

        self.arr_date = np.array(kospi.index)
        self.arr_open = np.array(kospi.Open, dtype=float)
        self.arr_close = np.array(kospi['Close'], dtype=float)
        self.arr_high = np.array(kospi.High, dtype=float)
        self.arr_low = np.array(kospi.Low, dtype=float)
        self.arr_volume = np.array(kospi.Volume, dtype=float)

if __name__ == "__main__":

    K = KOSPIDATA()

    FEATURES = ['high', 'low', 'open', 'close', 'volume']

    data = {'year': K.arr_date,
            'open': K.arr_open,
            'high': K.arr_high,
            'low': K.arr_low,
            'close': K.arr_close}

    df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close', 'volume'])

    dataset = df['close'].values
    print(dataset)

    # LSTM은 input data 단위에 민감함!, normalize 해줘야돼
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets(cross validation)
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    print(len(train), len(test))

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back = 2):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # reshape into X=t and Y=t+1
    look_back = 2
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim = look_back))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer ='adam')
    model.fit(trainX, trainY, nb_epoch = 100, batch_size=1, verbose =2)
    # verbose = 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch
    # verbose가 정확히 무엇일까..?

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
