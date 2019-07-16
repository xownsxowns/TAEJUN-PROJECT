
## P300 Classification
## CNN

# Epoch Sub1 ~ Sub30: TV
# Epoch Sub31 ~ Sub45: Doorlock
# Epoch Sub46 ~ Sub60: Lamp
# Epoch BS Sub 1 ~Sub45: Bluetooth speaker

# 1. Preprocessing
#  1) 0.5Hz highpass filter (FIR)
#  2) Bad channel rejection (1Hz lowpass filter , 2nd order Butter. , Corr. coeff < 0.4 , 70 % above)
#  3) Common average re-reference
#  4) 50Hz lowpass filter (FIR)
#  5) Artifact subspace reconstruction (cutoff: 10)
#
# 2. Data
#    ERP : [channel x time x stimulus type x block] (training: 50 block, test: 30 block)
#    target : [block x 1] target stimulus of each block


## validation, early stopping


from scipy import io
import pandas as pd
import numpy as np
import PIL.Image as pilimg
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

def define_model(input_shape):
    # channel 1
    input1 = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input1)
    batch1 = BatchNormalization()(conv1)
    acti1 = Activation('relu')(batch1)
    drop1 = Dropout(0.5)(acti1)
    conv1_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop1)
    batch1_2 = BatchNormalization()(conv1_2)
    acti1_2 = Activation('relu')(batch1_2)
    drop1_2 = Dropout(0.5)(acti1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop1_2)
    flat1 = Flatten()(pool1)
    # channel 2
    input2 = Input(shape=input_shape)
    conv2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input2)
    batch2 = BatchNormalization()(conv2)
    acti2 = Activation('relu')(batch2)
    drop2 = Dropout(0.5)(acti2)
    conv2_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop2)
    batch2_2 = BatchNormalization()(conv2_2)
    acti2_2 = Activation('relu')(batch2_2)
    drop2_2 = Dropout(0.5)(acti2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop2_2)
    flat2 = Flatten()(pool2)
    # channel 3
    input3 = Input(shape=input_shape)
    conv3 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input3)
    batch3 = BatchNormalization()(conv3)
    acti3 = Activation('relu')(batch3)
    drop3 = Dropout(0.5)(acti3)
    conv3_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop3)
    batch3_2 = BatchNormalization()(conv3_2)
    acti3_2 = Activation('relu')(batch3_2)
    drop3_2 = Dropout(0.5)(acti3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop3_2)
    flat3 = Flatten()(pool3)
    # channel 4
    input4 = Input(shape=input_shape)
    conv4 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input4)
    batch4 = BatchNormalization()(conv4)
    acti4 = Activation('relu')(batch4)
    drop4 = Dropout(0.5)(acti4)
    conv4_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop4)
    batch4_2 = BatchNormalization()(conv4_2)
    acti4_2 = Activation('relu')(batch4_2)
    drop4_2 = Dropout(0.5)(acti4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop4_2)
    flat4 = Flatten()(pool4)
    # channel 5
    input5 = Input(shape=input_shape)
    conv5 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input5)
    batch5 = BatchNormalization()(conv5)
    acti5 = Activation('relu')(batch5)
    drop5 = Dropout(0.5)(acti5)
    conv5_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop5)
    batch5_2 = BatchNormalization()(conv5_2)
    acti5_2 = Activation('relu')(batch5_2)
    drop5_2 = Dropout(0.5)(acti5_2)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop5_2)
    flat5 = Flatten()(pool5)
    # channel 6
    input6 = Input(shape=input_shape)
    conv6 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input6)
    batch6 = BatchNormalization()(conv6)
    acti6 = Activation('relu')(batch6)
    drop6 = Dropout(0.5)(acti6)
    conv6_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop6)
    batch6_2 = BatchNormalization()(conv6_2)
    acti6_2 = Activation('relu')(batch6_2)
    drop6_2 = Dropout(0.5)(acti6_2)
    pool6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop6_2)
    flat6 = Flatten()(pool6)
    # channel 7
    input7 = Input(shape=input_shape)
    conv7 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input7)
    batch7 = BatchNormalization()(conv7)
    acti7 = Activation('relu')(batch7)
    drop7 = Dropout(0.5)(acti7)
    conv7_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop7)
    batch7_2 = BatchNormalization()(conv7_2)
    acti7_2 = Activation('relu')(batch7_2)
    drop7_2 = Dropout(0.5)(acti7_2)
    pool7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop7_2)
    flat7 = Flatten()(pool7)
    # channel 8
    input8 = Input(shape=input_shape)
    conv8 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input8)
    batch8 = BatchNormalization()(conv8)
    acti8 = Activation('relu')(batch8)
    drop8 = Dropout(0.5)(acti8)
    conv8_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop8)
    batch8_2 = BatchNormalization()(conv8_2)
    acti8_2 = Activation('relu')(batch8_2)
    drop8_2 = Dropout(0.5)(acti8_2)
    pool8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop8_2)
    flat8 = Flatten()(pool8)
    # channel 9
    input9 = Input(shape=input_shape)
    conv9 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input9)
    batch9 = BatchNormalization()(conv9)
    acti9 = Activation('relu')(batch9)
    drop9 = Dropout(0.5)(acti9)
    conv9_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop9)
    batch9_2 = BatchNormalization()(conv9_2)
    acti9_2 = Activation('relu')(batch9_2)
    drop9_2 = Dropout(0.5)(acti9_2)
    pool9 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop9_2)
    flat9 = Flatten()(pool9)
    # channel 10
    input10 = Input(shape=input_shape)
    conv10 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input10)
    batch10 = BatchNormalization()(conv10)
    acti10 = Activation('relu')(batch10)
    drop10 = Dropout(0.5)(acti10)
    conv10_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop10)
    batch10_2 = BatchNormalization()(conv10_2)
    acti10_2 = Activation('relu')(batch10_2)
    drop10_2 = Dropout(0.5)(acti10_2)
    pool10 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop10_2)
    flat10 = Flatten()(pool10)
    # channel 11
    input11 = Input(shape=input_shape)
    conv11 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input11)
    batch11 = BatchNormalization()(conv11)
    acti11 = Activation('relu')(batch11)
    drop11 = Dropout(0.5)(acti11)
    conv11_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop11)
    batch11_2 = BatchNormalization()(conv11_2)
    acti11_2 = Activation('relu')(batch11_2)
    drop11_2 = Dropout(0.5)(acti11_2)
    pool11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop11_2)
    flat11 = Flatten()(pool11)
    # channel 12
    input12 = Input(shape=input_shape)
    conv12 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input12)
    batch12 = BatchNormalization()(conv12)
    acti12 = Activation('relu')(batch12)
    drop12 = Dropout(0.5)(acti12)
    conv12_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop12)
    batch12_2 = BatchNormalization()(conv12_2)
    acti12_2 = Activation('relu')(batch12_2)
    drop12_2 = Dropout(0.5)(acti12_2)
    pool12 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop12_2)
    flat12 = Flatten()(pool12)
    # channel 13
    input13 = Input(shape=input_shape)
    conv13 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input13)
    batch13 = BatchNormalization()(conv13)
    acti13 = Activation('relu')(batch13)
    drop13 = Dropout(0.5)(acti13)
    conv13_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop13)
    batch13_2 = BatchNormalization()(conv13_2)
    acti13_2 = Activation('relu')(batch13_2)
    drop13_2 = Dropout(0.5)(acti13_2)
    pool13 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop13_2)
    flat13 = Flatten()(pool13)
    # channel 14
    input14 = Input(shape=input_shape)
    conv14 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input14)
    batch14 = BatchNormalization()(conv14)
    acti14 = Activation('relu')(batch14)
    drop14 = Dropout(0.5)(acti14)
    conv14_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop14)
    batch14_2 = BatchNormalization()(conv14_2)
    acti14_2 = Activation('relu')(batch14_2)
    drop14_2 = Dropout(0.5)(acti14_2)
    pool14 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop14_2)
    flat14 = Flatten()(pool14)
    # channel 15
    input15 = Input(shape=input_shape)
    conv15 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input15)
    batch15 = BatchNormalization()(conv15)
    acti15 = Activation('relu')(batch15)
    drop15 = Dropout(0.5)(acti15)
    conv15_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop15)
    batch15_2 = BatchNormalization()(conv15_2)
    acti15_2 = Activation('relu')(batch15_2)
    drop15_2 = Dropout(0.5)(acti15_2)
    pool15 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop15_2)
    flat15 = Flatten()(pool15)
    # channel 1
    input16 = Input(shape=input_shape)
    conv16 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input16)
    batch16 = BatchNormalization()(conv16)
    acti16 = Activation('relu')(batch16)
    drop16 = Dropout(0.5)(acti16)
    conv16_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop16)
    batch16_2 = BatchNormalization()(conv16_2)
    acti16_2 = Activation('relu')(batch16_2)
    drop16_2 = Dropout(0.5)(acti16_2)
    pool16 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop16_2)
    flat16 = Flatten()(pool16)
    # channel 17
    input17 = Input(shape=input_shape)
    conv17 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input17)
    batch17 = BatchNormalization()(conv17)
    acti17 = Activation('relu')(batch17)
    drop17 = Dropout(0.5)(acti17)
    conv17_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop17)
    batch17_2 = BatchNormalization()(conv17_2)
    acti17_2 = Activation('relu')(batch17_2)
    drop17_2 = Dropout(0.5)(acti17_2)
    pool17 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop17_2)
    flat17 = Flatten()(pool17)
    # channel 18
    input18 = Input(shape=input_shape)
    conv18 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input18)
    batch18 = BatchNormalization()(conv18)
    acti18 = Activation('relu')(batch18)
    drop18 = Dropout(0.5)(acti18)
    conv18_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop18)
    batch18_2 = BatchNormalization()(conv18_2)
    acti18_2 = Activation('relu')(batch18_2)
    drop18_2 = Dropout(0.5)(acti18_2)
    pool18 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop18_2)
    flat18 = Flatten()(pool18)
    # channel 19
    input19 = Input(shape=input_shape)
    conv19 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input19)
    batch19 = BatchNormalization()(conv19)
    acti19 = Activation('relu')(batch19)
    drop19 = Dropout(0.5)(acti19)
    conv19_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop19)
    batch19_2 = BatchNormalization()(conv19_2)
    acti19_2 = Activation('relu')(batch19_2)
    drop19_2 = Dropout(0.5)(acti19_2)
    pool19 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop19_2)
    flat19 = Flatten()(pool19)
    # channel 20
    input20 = Input(shape=input_shape)
    conv20 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input20)
    batch20 = BatchNormalization()(conv20)
    acti20 = Activation('relu')(batch20)
    drop20 = Dropout(0.5)(acti20)
    conv20_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop20)
    batch20_2 = BatchNormalization()(conv20_2)
    acti20_2 = Activation('relu')(batch20_2)
    drop20_2 = Dropout(0.5)(acti20_2)
    pool20 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop20_2)
    flat20 = Flatten()(pool20)
    # channel 21
    input21 = Input(shape=input_shape)
    conv21 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input21)
    batch21 = BatchNormalization()(conv21)
    acti21 = Activation('relu')(batch21)
    drop21 = Dropout(0.5)(acti21)
    conv21_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop21)
    batch21_2 = BatchNormalization()(conv21_2)
    acti21_2 = Activation('relu')(batch21_2)
    drop21_2 = Dropout(0.5)(acti21_2)
    pool21 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop21_2)
    flat21 = Flatten()(pool21)
    # channel 22
    input22 = Input(shape=input_shape)
    conv22 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input22)
    batch22 = BatchNormalization()(conv22)
    acti22 = Activation('relu')(batch22)
    drop22 = Dropout(0.5)(acti22)
    conv22_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop22)
    batch22_2 = BatchNormalization()(conv22_2)
    acti22_2 = Activation('relu')(batch22_2)
    drop22_2 = Dropout(0.5)(acti22_2)
    pool22 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop22_2)
    flat22 = Flatten()(pool22)
    # channel 23
    input23 = Input(shape=input_shape)
    conv23 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input23)
    batch23 = BatchNormalization()(conv23)
    acti23 = Activation('relu')(batch23)
    drop23 = Dropout(0.5)(acti23)
    conv23_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop23)
    batch23_2 = BatchNormalization()(conv23_2)
    acti23_2 = Activation('relu')(batch23_2)
    drop23_2 = Dropout(0.5)(acti23_2)
    pool23 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop23_2)
    flat23 = Flatten()(pool23)
    # channel 24
    input24 = Input(shape=input_shape)
    conv24 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input24)
    batch24 = BatchNormalization()(conv24)
    acti24 = Activation('relu')(batch24)
    drop24 = Dropout(0.5)(acti24)
    conv24_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop24)
    batch24_2 = BatchNormalization()(conv24_2)
    acti24_2 = Activation('relu')(batch24_2)
    drop24_2 = Dropout(0.5)(acti24_2)
    pool24 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop24_2)
    flat24 = Flatten()(pool24)
    # channel 25
    input25 = Input(shape=input_shape)
    conv25 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input25)
    batch25 = BatchNormalization()(conv25)
    acti25 = Activation('relu')(batch25)
    drop25 = Dropout(0.5)(acti25)
    conv25_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop25)
    batch25_2 = BatchNormalization()(conv25_2)
    acti25_2 = Activation('relu')(batch25_2)
    drop25_2 = Dropout(0.5)(acti25_2)
    pool25 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop25_2)
    flat25 = Flatten()(pool25)
    # channel 26
    input26 = Input(shape=input_shape)
    conv26 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input26)
    batch26 = BatchNormalization()(conv26)
    acti26 = Activation('relu')(batch26)
    drop26 = Dropout(0.5)(acti26)
    conv26_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop26)
    batch26_2 = BatchNormalization()(conv26_2)
    acti26_2 = Activation('relu')(batch26_2)
    drop26_2 = Dropout(0.5)(acti26_2)
    pool26 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop26_2)
    flat26 = Flatten()(pool26)
    # channel 27
    input27 = Input(shape=input_shape)
    conv27 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input27)
    batch27 = BatchNormalization()(conv27)
    acti27 = Activation('relu')(batch27)
    drop27 = Dropout(0.5)(acti27)
    conv27_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop27)
    batch27_2 = BatchNormalization()(conv27_2)
    acti27_2 = Activation('relu')(batch27_2)
    drop27_2 = Dropout(0.5)(acti27_2)
    pool27 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop27_2)
    flat27 = Flatten()(pool27)
    # channel 28
    input28 = Input(shape=input_shape)
    conv28 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input28)
    batch28 = BatchNormalization()(conv28)
    acti28 = Activation('relu')(batch28)
    drop28 = Dropout(0.5)(acti28)
    conv28_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop28)
    batch28_2 = BatchNormalization()(conv28_2)
    acti28_2 = Activation('relu')(batch28_2)
    drop28_2 = Dropout(0.5)(acti28_2)
    pool28 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop28_2)
    flat28 = Flatten()(pool28)
    # channel 29
    input29 = Input(shape=input_shape)
    conv29 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input29)
    batch29 = BatchNormalization()(conv29)
    acti29 = Activation('relu')(batch29)
    drop29 = Dropout(0.5)(acti29)
    conv29_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop29)
    batch29_2 = BatchNormalization()(conv29_2)
    acti29_2 = Activation('relu')(batch29_2)
    drop29_2 = Dropout(0.5)(acti29_2)
    pool29 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop29_2)
    flat29 = Flatten()(pool29)
    # channel 30
    input30 = Input(shape=input_shape)
    conv30 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(input30)
    batch30 = BatchNormalization()(conv30)
    acti30 = Activation('relu')(batch30)
    drop30 = Dropout(0.5)(acti30)
    conv30_2 = Conv2D(32, (3, 3), input_shape=input_shape, padding='same')(drop30)
    batch30_2 = BatchNormalization()(conv30_2)
    acti30_2 = Activation('relu')(batch30_2)
    drop30_2 = Dropout(0.5)(acti30_2)
    pool30 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop30_2)
    flat30 = Flatten()(pool30)
    # merge
    merged = concatenate([flat1,flat2,flat3,flat4,flat5,flat6,flat7,flat8,flat9,flat10,
                          flat11,flat12,flat13,flat14,flat15,flat16,flat17,flat18,flat19,flat20,
                          flat21,flat22,flat23,flat24,flat25,flat26,flat27,flat28,flat29,flat30])
    # interpretation
    dense1 = Dense(50, activation='tanh')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,
                          input11,input12,input13,input14,input15,input16,input17,input18,input19,input20,
                          input21,input22,input23,input24,input25,input26,input27,input28,input29,input30], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

total_acc = list()
isub = 0
print(isub)
path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
data = io.loadmat(path)

nch = np.shape(data['ERP'])[0]
nlen = 250
ntrain = np.shape(data['ERP'])[3]

tar_data = list()
tar_label = list()
nontar_data = list()
nontar_label = list()

# 100ms~600ms 길이 자른것
for i in range(ntrain):
    target = data['ERP'][:, 150:, data['target'][i][0] - 1, i]
    tar_data.append(target)
    tar_label.append(1)

    for j in range(4):
        if j == (data['target'][i][0] - 1):
            continue
        else:
            nontar_data.append(data['ERP'][:, 150:, j, i])
            nontar_label.append(0)

tar_data = np.reshape(tar_data, (ntrain, nlen, nch))
nontar_data = np.reshape(nontar_data, ((ntrain * 3), nlen, nch))

nontar_trial = list()
# non target data
for itrial in range(nontar_data.shape[0]):
    for ich in range(nontar_data.shape[2]):
        file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub + 1) + '/nontar_trial' + str(itrial + 1) + '_GAF_ch' + str(ich + 1) + '.png'
        if ich == 0:
            image_open = pilimg.open(file_path)
            image_open = image_open.resize((64,64))
            image = np.array(image_open)[...,:3]
            image_total = image
        else:
            image_open = pilimg.open(file_path)
            image_open = image_open.resize((64, 64))
            image = np.array(image_open)[...,:3]
            image_total = np.concatenate((image_total, image), axis=2)
    nontar_trial.append(image_total)
    print('{0} nontar trial ended'.format(itrial+1))

tar_trial = list()
# target data
for itrial in range(tar_data.shape[0]):
    for ich in range(tar_data.shape[2]):
        file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub + 1) + '/tar_trial' + str(itrial + 1) + '_GAF_ch' + str(ich + 1) + '.png'
        if ich == 0:
            image_open = pilimg.open(file_path)
            image_open = image_open.resize((64,64))
            image = np.array(image_open)[...,:3]
            image_total = image
        else:
            image_open = pilimg.open(file_path)
            image_open = image_open.resize((64, 64))
            image = np.array(image_open)[...,:3]
            image_total = np.concatenate((image_total, image), axis=2)
    tar_trial.append(image_total)
    print('{0} tar trial ended'.format(itrial+1))

train_vali_data = np.concatenate((tar_trial, nontar_trial))
train_vali_label = np.concatenate((tar_label, nontar_label))

train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.15, random_state=42)
nch = tar_data.shape[2]

# print(model.summary())
# early_stopping = EarlyStopping(patience=20)
# model.fit(train_data, train_label, epochs=500, batch_size=20, validation_data=(vali_data, vali_label), callbacks=[early_stopping])
input_shape = (64, 64, 3)
early_stopping = EarlyStopping(patience=20)
model = define_model(input_shape=input_shape)
model.fit([train_data[...,0:3],train_data[...,3:6],train_data[...,6:9],train_data[...,9:12],train_data[...,12:15],train_data[...,15:18],
            train_data[...,18:21],train_data[...,21:24],train_data[...,24:27],train_data[...,27:30],train_data[...,30:33],train_data[...,33:36],
            train_data[...,36:39],train_data[...,39:42],train_data[...,42:45],train_data[...,45:48],train_data[...,48:51],train_data[...,51:54],
            train_data[...,54:57],train_data[...,57:60],train_data[...,60:63],train_data[...,63:66],train_data[...,66:69],train_data[...,69:72],train_data[...,72:75],
            train_data[...,75:78],train_data[...,78:81],train_data[...,81:84],train_data[...,84:87],train_data[...,87:90]], train_label, epochs=500, batch_size=20
          , validation_data=(vali_data, vali_label), callbacks=[early_stopping])
## Test
path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
data = io.loadmat(path)

nch = np.shape(data['ERP'])[0]
nlen = 250
ntest = np.shape(data['ERP'])[3]
nstim = 4

test_data = list()
# ntest, nch, nlen, nstim
# 100ms~600ms 길이 자른것
for i in range(ntest):
    target = data['ERP'][:, 150:, :, i]
    test_data.append(target)
# nstim, ntest, nlen, nch
test_data = np.transpose(test_data, (3, 0, 2, 1))
corr_ans = 0

for itrial in range(test_data.shape[1]):
    stim1_trial = list()
    stim2_trial = list()
    stim3_trial = list()
    stim4_trial = list()
    total_prob = list()
    for nstim in range(test_data.shape[0]):
        for ich in range(tar_data.shape[2]):
            file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub + 1) + '/test_trial' + str(nstim+1) + '-' + str(itrial+1) + '_GAF_ch' + str(ich+1) + '.png'
            if ich == 0:
                image_open = pilimg.open(file_path)
                image_open = image_open.resize((64,64))
                image = np.array(image_open)[...,:3]
                image_total = image
            else:
                image_open = pilimg.open(file_path)
                image_open = image_open.resize((64, 64))
                image = np.array(image_open)[...,:3]
                image_total = np.concatenate((image_total, image), axis=2)
        if nstim == 0:
            stim1_trial.append(image_total)
        elif nstim == 1:
            stim2_trial.append(image_total)
        elif nstim == 2:
            stim3_trial.append(image_total)
        elif nstim == 3:
            stim4_trial.append(image_total)

    total_trial = np.concatenate((stim1_trial,stim2_trial,stim3_trial,stim4_trial))
    prob = model.predict(total_trial)
    total_prob.append(prob[0][0])
    total_prob.append(prob[1][0])
    total_prob.append(prob[2][0])
    total_prob.append(prob[3][0])

    predicted_label = np.argmax(total_prob)
    if data['target'][itrial][0] == (predicted_label+1):
        corr_ans += 1

    print('sub{0}: {1} test trial ended'.format(isub+1, itrial+1))

total_acc.append((corr_ans/ntest)*100)
print("Accuracy: %.2f%%" % ((corr_ans/ntest)*100))
print(total_acc)
print(np.mean(total_acc))

df = pd.DataFrame(total_acc)
filename = 'P300_Result_CNN_GAF.csv'
df.to_csv(filename)