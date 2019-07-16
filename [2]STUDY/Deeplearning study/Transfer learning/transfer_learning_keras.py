import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

# mobilenet have a last layer consisting of 1000 neurons (one for each class)
# we discard the 1000 neuron layer and add our own last layer for the network.
# setting (IncludeTop=False) when importing the model.

base_model = MobileNet(weights='imagenet', include_top=False)
# imports the mobilenet model and discards the last 1000 neuron layers.
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) # we add dense layers so that the model can learn more complex functions and classifiy for better results.
x = Dense(1024, activation='relu')(x) # dense layer2
x = Dense(512, activation='relu')(x) # dense layer3
preds = Dense(120, activation='softmax')(x) # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# # to set all the weights to be non-trainable
# for layer in model.layers:
#     layer.trainable = False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True




