from scipy import io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM, SimpleRNN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
path = 'J:/bike-sharing-dataset/data_regression.mat'
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

from keras.models import load_model
model = load_model('epoch500_8.h5')

y_pred = model.predict(test_X)
RNN_RMSE = np.sqrt(np.mean((test_Y - y_pred)**2))
print(RNN_RMSE)

df = pd.DataFrame(y_pred)
filename = 'y_pred.csv'
df.to_csv(filename)