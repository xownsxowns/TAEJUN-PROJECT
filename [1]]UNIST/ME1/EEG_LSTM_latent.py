import os
import numpy as np
from scipy import io
# import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)

## EEG 데이터 불러오기
data = io.loadmat('/Users/Taejun/Documents/GitHub/Python_project/[1]]UNIST/ME1/ME1.mat')
eeg = data['data'][:,150:,:]
eeg = np.transpose(eeg, (1,0,2))
n_ch = 29
n_timepoint = 1000
## Learning EEG latent space
model = Sequential()
model.add(LSTM(50, input_shape=(n_timepoint,n_ch)))
model.add(Dense(n_ch, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



