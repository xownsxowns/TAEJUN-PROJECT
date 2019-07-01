import pandas as pd
import numpy as np
import random
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
model = load_model('C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/Face and object detection/model2.h5')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('E:/[2] 연구/[3] Facial/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='categorical')

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_set, steps=(3000 // 32+1 ))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
class_name = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print(classification_report(test_set.classes, y_pred, target_names=class_name))
