from scipy import io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM, SimpleRNN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
path = '/Users/Taejun/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/Multivariate analysis/data_regression.mat'
data = io.loadmat(path)

X_sub = data['X_sub']
Y = data['Y']

## training: 0~697
## test:697~end

train_X = X_sub[0:697,:]
train_X = np.expand_dims(train_X, axis=2)
train_Y = Y[0:697,:]
test_X = X_sub[697:,:]
test_X = np.expand_dims(test_X, axis=2)
test_Y = Y[697:,:]

model = Sequential()
model.add(SimpleRNN(10, input_shape=(10,1), return_sequences=True))
model.add(SimpleRNN(10))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(train_X, train_Y, epochs=500, batch_size=8, verbose=1)
model.save('epoch500_8.h5')

y_pred = model.predict(test_X)
RNN_RMSE = np.sqrt(np.mean((test_Y - y_pred)**2))


plt.plot(test_Y, color='#0072BD')
plt.plot(y_pred, color='#A2142F')
plt.title('RNN')
plt.legend(['Y','Y predict'])
plt.show()