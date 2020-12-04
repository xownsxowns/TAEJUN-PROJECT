import os
import numpy as np
from scipy import io
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from sklearn.preprocessing import StandardScaler

## 각 EEG latent를 for문에서 뽑아
os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)
random_dim = 100
## EEG 데이터 불러오기
# data = io.loadmat('/Users/Taejun/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/ME1.mat')
data = io.loadmat('C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/ME1.mat')
eeg = data['data'][:,150:,:]
eeg = np.transpose(eeg, (1,0,2))
n_ch = 29
n_timepoint = 1000

## MNIST 데이터 불러오기
# mnist_data = io.loadmat('/Users/Taejun/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/data.mat')
mnist_data = io.loadmat('C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/data.mat')
mnist_total = mnist_data['data'][0]


# label 불러오기
# label = pd.read_csv('/Users/Taejun/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/label.txt',header=None, engine='python')
label_data = pd.read_csv('C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[1]]UNIST/ME1/label.txt',header=None, engine='python')
label_data.columns = ['label']
label_data.head()

mnist_data_total = list()
for i in range(len(mnist_total)):
    mnist_data_total.append(mnist_total[i])
mnist_data_total = np.reshape(mnist_data_total, (500, (np.shape(mnist_data_total)[1] * np.shape(mnist_data_total)[2])))

for i in range(10):
    # label이 1인 index 찾기
    index = label_data[label_data['label']==i].index
    label = np.ones((np.shape(index))) * i
    # label이 1인 eeg 데이터 찾기
    eeg_picked = eeg[:,:,index]
    eeg_data = np.reshape(eeg_picked,(np.shape(eeg_picked)[2],1000,29))

    ## Learning EEG latent space
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_timepoint,n_ch)))
    model.add(Dense(n_ch, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(eeg_data, label, epochs=100, batch_size=20, verbose=2)

    # with a Sequential model
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[1].output])
    layer_output = get_3rd_layer_output([eeg_data])[0]
    np.shape(layer_output)
    # layer_output is (66,29)

    # latent variable standardization
    latent = StandardScaler().fit_transform(np.transpose(layer_output)).transpose()

    # GAN
    # Adam optimizer를 사용
    def get_optimizer():
        return Adam(lr=0.0002, beta_1=0.5)


    # Generator 만들기
    def get_generator(optimizer):
        generator = Sequential()
        generator.add(Dense(256, input_dim=(n_ch+random_dim),
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return generator


    # Discriminator 만들기
    def get_discriminator(optimizer):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784,
                                kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator


    # random_dim 대신에 EEG dimension 넣어보자
    def get_gan_network(discriminator, random_dim, generator, optimizer, n_ch):
        # 우리는 generator와 discriminator를 동시에 학습시키고 싶을 때 trainable을 False로 설정합니다.
        discriminator.trainable = False

        # GAN 입력 (노이즈)은 위에서 100 차원으로 설정했습니다. (EEG는 다를듯)
        gan_input = Input(shape=((random_dim+n_ch),))

        # Generator의 결과는 이미지 입니다.
        x = generator(gan_input)

        # Discriminator의 결과는 이미지가 진짜인지 가짜인지에 대한 확률입니다.
        gan_output = discriminator(x)

        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        return gan

    # 생성된 MNIST 이미지를 보여주는 함수
    def plot_generated_images(epoch, mnist_num, generator, latent, examples=1, dim=(1,1), figsize=(10,10)):
        noise = np.random.normal(0,1,size=[examples, random_dim])
        noise_eeg = np.concatenate((noise, latent[:examples,:]), axis=1)
        generated_images = generator.predict(noise_eeg)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for k in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], k+1)
            plt.imshow(generated_images[k], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('gan_generated_image_' + str(mnist_num) + '_epoch_%d.png' % epoch)


    # 네트워크 훈련 및 이미지 확인
    epochs=200
    batch_size=np.shape(eeg_data)[0]
    # train 데이터와 test 데이터를 가져옵니다.
    x_train = mnist_data_total[index,:]

    # train 데이터를 128 사이즈의 batch 로 나눕니다.
    batch_count = x_train.shape[0] // batch_size

    # 우리의 GAN 네트워크를 만듭니다.
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam, n_ch)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15, '\n')
        for _ in tqdm(range(batch_count)):
            # 입력으로 사용할 random 노이즈와 이미지를 가져옵니다.
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            noise_eeg = np.concatenate((noise, latent[:batch_size,:]), axis=1)
            image_batch = x_train[:batch_size,:]

            # MNIST 이미지를 생성합니다
            generated_images = generator.predict(noise_eeg)
            X = np.concatenate([image_batch, generated_images])

            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # Discriminator를 학습시킵니다.
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Generator를 학습시킵니다.
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            noise_eeg = np.concatenate((noise, latent[:batch_size,:]), axis=1)
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise_eeg, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, i, generator, latent)