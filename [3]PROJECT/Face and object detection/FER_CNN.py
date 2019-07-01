import pandas as pd
import numpy as np
import random
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation

## CNN 모델로 해보자 유명한걸로
# Initialize the CNN
model = Sequential()
# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(64,64,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# Step 3 - Flattening
model.add(Flatten())
# Step 4 - Full connection
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
print(model.summary())
# Complie the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

# Fitting the CNN to the images
# class 0:Anger, 1:Fear, 2:Happy, 3:Neutral, 4:Sad, 5:Surprise
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('E:/[2] 연구/[3] Facial/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('E:/[2] 연구/[3] Facial/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='categorical')

model.fit_generator(training_set,
                    steps_per_epoch=(280098//32),
                    epochs=50,
                    validation_data=test_set,
                    validation_steps=20)


model.save("model2.h5")
print("Saved model to disk")