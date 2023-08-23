from keras import layers, models, optimizers
from keras import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

K.set_image_data_format('channels_first')
print(K.image_data_format())


class GAN(models.Sequential):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

        self.generator = self.make_G()
        self.discriminator = self.make_D()

        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)

        self.compile_all()

    def make_G(self):
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def make_D(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh',
                                input_shape=(1, 28, 28)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), padding='same', activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def compile_all(self):
        opt_D = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        opt_G = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)

        self.compile(loss='binary_crossentropy', optimizer=opt_G)

        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt_D)

    def get_z(self, ln):
        return np.random.uniform(-1.0, 1.0, (ln, self.input_dim))

    def train_once(self, x):
        ln = x.shape[0]

        z = self.get_z(ln)
        gen = self.generator.predict(z, verbose=0)
        input_D = np.concatenate((x, gen))
        y_D = [1] * ln + [0] * ln
        loss_D = self.discriminator.train_on_batch(input_D, y_D)

        z = self.get_z(ln)
        self.discriminator.trainable = False
        loss_G = self.train_on_batch(z, [1] * ln)
        self.discriminator.trainable = True

        return loss_D, loss_G


def get_x(x_train, index, batch_size):
    return x_train[index * batch_size:(index + 1) * batch_size]


class MnistData():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        img_rows, img_cols = x_train.shape[1:]

        x_train = x_train.astype('float32') - 127.5
        x_test = x_test.astype('float32') - 127.5
        x_train /= 127.5
        x_test /= 127.5

        self.num_classes = 10
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


def main():
    batch_size = 100
    epochs = 30
    input_dim = 100
    sample_size = 6

    data = MnistData()
    x_train = data.x_train
    x_train = x_train.reshape((x_train.shape[0], 1) + x_train.shape[1:])

    gan = GAN(input_dim)

    for epoch in range(epochs):
        print("Epoch", epoch)

        for index in range(int(x_train.shape[0] / batch_size)):
            x = get_x(x_train, index, batch_size)
            loss_D, loss_G = gan.train_once(x)

        print('Loss D:', loss_D)
        print('Loss G:', loss_G)

        if epoch % 2 == 0 or epoch == epochs - 1:
            z = gan.get_z(sample_size)
            gen = gan.generator.predict(z, verbose=0)

            plt.figure(figsize=(20, 2))

            for i in range(sample_size):
                ax = plt.subplot(1, sample_size, i + 1)
                plt.imshow(gen[i].reshape((28, 28)))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.show()


if __name__ == '__main__':
    main()
