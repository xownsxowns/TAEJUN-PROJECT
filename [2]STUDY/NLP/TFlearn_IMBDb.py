import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDb 데이터셋 로딩
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=100000, valid_portion=0.1)

X_train, Y_train = train
X_test, Y_test = test

# 최대 시퀀스 길이를 100으로 하여 tflearn.data_utils.pad_sequences ()를 사용해 제로 패딩 형태로 시퀀스 길이를 맞춘다.
X_train = pad_sequences(X_train, maxlen=100, value=0.)
X_test = pad_sequences(X_test, maxlen=100, value=0.)
Y_train = to_categorical(Y_train, nb_classes=2)
Y_test = to_categorical(Y_test, nb_classes=2)

RNN = tflearn.input_data([None, 100])
RNN = tflearn.embedding(RNN, input_dim=10000, output_dim=128)
# 마지막으로 LSTM 계층과 완전 연결 계층을 추가하여 이진 결과 (좋거나, 나쁘거나)를 출력한다.

RNN = tflearn.lstm(RNN, 128, dropout=0.8)
RNN = tflearn.fully_connected(RNN, 2, activation='softmax')
RNN = tflearn.regression(RNN, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

# 네트워크 학습
model = tflearn.DNN(RNN, tensorboard_verbose=0)
model.fit(X_train, Y_train, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=32)