from scipy import io, signal
import pandas as pd
import numpy as np
import random
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from imblearn.under_sampling import *
from sklearn.metrics import accuracy_score
import gc
import keras.backend as K

# parameter setting

np.random.seed(0)
random.seed(0)


def convert_to_2d_doorlock_light(sub_num, input):
    ch_path1 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/ch/chlist_doorlock'
    ch_path2 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/ch/chlist_light'
    ch_data1 = io.loadmat(ch_path1)
    ch_data2 = io.loadmat(ch_path2)
    ch_list = []
    for i in range(len(ch_data1['chlist_doorlock'][0])):
        ch_list.append(ch_data1['chlist_doorlock'][0][i][0])
    for ii in range(len(ch_data2['chlist_light'][0])):
        ch_list.append(ch_data2['chlist_light'][0][ii][0])
    sub_ch_list = ch_list[sub_num - 30]
    mapp = np.zeros((np.shape(input)[0], 7, 11, np.shape(input)[2]))
    for itrial in range(np.shape(input)[0]):
        for timepoint in range(np.shape(input)[2]):
            for a in range(np.shape(input)[1]):
                if 1 in sub_ch_list:
                    mapp[itrial, 0, 3, timepoint] = input[itrial][np.where(sub_ch_list == 1)[0][0], timepoint]
                if 2 in sub_ch_list:
                    mapp[itrial, 0, 5, timepoint] = input[itrial][np.where(sub_ch_list == 2)[0][0], timepoint]
                if 3 in sub_ch_list:
                    mapp[itrial, 0, 7, timepoint] = input[itrial][np.where(sub_ch_list == 3)[0][0], timepoint]
                if 4 in sub_ch_list:
                    mapp[itrial, 1, 1, timepoint] = input[itrial][np.where(sub_ch_list == 4)[0][0], timepoint]
                if 5 in sub_ch_list:
                    mapp[itrial, 1, 3, timepoint] = input[itrial][np.where(sub_ch_list == 5)[0][0], timepoint]
                if 6 in sub_ch_list:
                    mapp[itrial, 1, 5, timepoint] = input[itrial][np.where(sub_ch_list == 6)[0][0], timepoint]
                if 7 in sub_ch_list:
                    mapp[itrial, 1, 7, timepoint] = input[itrial][np.where(sub_ch_list == 7)[0][0], timepoint]
                if 8 in sub_ch_list:
                    mapp[itrial, 1, 9, timepoint] = input[itrial][np.where(sub_ch_list == 8)[0][0], timepoint]
                if 9 in sub_ch_list:
                    mapp[itrial, 2, 0, timepoint] = input[itrial][np.where(sub_ch_list == 9)[0][0], timepoint]
                if 10 in sub_ch_list:
                    mapp[itrial, 2, 2, timepoint] = input[itrial][np.where(sub_ch_list == 10)[0][0], timepoint]
                if 11 in sub_ch_list:
                    mapp[itrial, 2, 4, timepoint] = input[itrial][np.where(sub_ch_list == 11)[0][0], timepoint]
                if 12 in sub_ch_list:
                    mapp[itrial, 2, 6, timepoint] = input[itrial][np.where(sub_ch_list == 12)[0][0], timepoint]
                if 13 in sub_ch_list:
                    mapp[itrial, 2, 8, timepoint] = input[itrial][np.where(sub_ch_list == 13)[0][0], timepoint]
                if 14 in sub_ch_list:
                    mapp[itrial, 2, 10, timepoint] = input[itrial][np.where(sub_ch_list == 14)[0][0], timepoint]
                if 15 in sub_ch_list:
                    mapp[itrial, 3, 1, timepoint] = input[itrial][np.where(sub_ch_list == 15)[0][0], timepoint]
                if 16 in sub_ch_list:
                    mapp[itrial, 3, 3, timepoint] = input[itrial][np.where(sub_ch_list == 16)[0][0], timepoint]
                if 17 in sub_ch_list:
                    mapp[itrial, 3, 5, timepoint] = input[itrial][np.where(sub_ch_list == 17)[0][0], timepoint]
                if 18 in sub_ch_list:
                    mapp[itrial, 3, 7, timepoint] = input[itrial][np.where(sub_ch_list == 18)[0][0], timepoint]
                if 19 in sub_ch_list:
                    mapp[itrial, 3, 9, timepoint] = input[itrial][np.where(sub_ch_list == 19)[0][0], timepoint]
                if 20 in sub_ch_list:
                    mapp[itrial, 4, 2, timepoint] = input[itrial][np.where(sub_ch_list == 20)[0][0], timepoint]
                if 21 in sub_ch_list:
                    mapp[itrial, 4, 4, timepoint] = input[itrial][np.where(sub_ch_list == 21)[0][0], timepoint]
                if 22 in sub_ch_list:
                    mapp[itrial, 4, 6, timepoint] = input[itrial][np.where(sub_ch_list == 22)[0][0], timepoint]
                if 23 in sub_ch_list:
                    mapp[itrial, 4, 8, timepoint] = input[itrial][np.where(sub_ch_list == 23)[0][0], timepoint]
                if 24 in sub_ch_list:
                    mapp[itrial, 5, 1, timepoint] = input[itrial][np.where(sub_ch_list == 24)[0][0], timepoint]
                if 25 in sub_ch_list:
                    mapp[itrial, 5, 3, timepoint] = input[itrial][np.where(sub_ch_list == 25)[0][0], timepoint]
                if 26 in sub_ch_list:
                    mapp[itrial, 5, 5, timepoint] = input[itrial][np.where(sub_ch_list == 26)[0][0], timepoint]
                if 27 in sub_ch_list:
                    mapp[itrial, 5, 7, timepoint] = input[itrial][np.where(sub_ch_list == 27)[0][0], timepoint]
                if 28 in sub_ch_list:
                    mapp[itrial, 5, 9, timepoint] = input[itrial][np.where(sub_ch_list == 28)[0][0], timepoint]
                if 29 in sub_ch_list:
                    mapp[itrial, 6, 3, timepoint] = input[itrial][np.where(sub_ch_list == 29)[0][0], timepoint]
                if 30 in sub_ch_list:
                    mapp[itrial, 6, 5, timepoint] = input[itrial][np.where(sub_ch_list == 30)[0][0], timepoint]
                if 31 in sub_ch_list:
                    mapp[itrial, 6, 7, timepoint] = input[itrial][np.where(sub_ch_list == 31)[0][0], timepoint]
    return mapp


def convert_to_2d_bs(sub_num, input):
    ch_path1 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/ch/chlist_bs'
    ch_data1 = io.loadmat(ch_path1)
    ch_list = []
    for i in range(len(ch_data1['chlist_bs'][0])):
        ch_list.append(ch_data1['chlist_bs'][0][i][0])
    sub_ch_list = ch_list[sub_num]
    mapp = np.zeros((np.shape(input)[0], 7, 11, np.shape(input)[2]))
    for itrial in range(np.shape(input)[0]):
        for timepoint in range(np.shape(input)[2]):
            for a in range(np.shape(input)[1]):
                if 1 in sub_ch_list:
                    mapp[itrial, 0, 3, timepoint] = input[itrial][np.where(sub_ch_list == 1)[0][0], timepoint]
                if 2 in sub_ch_list:
                    mapp[itrial, 0, 5, timepoint] = input[itrial][np.where(sub_ch_list == 2)[0][0], timepoint]
                if 3 in sub_ch_list:
                    mapp[itrial, 0, 7, timepoint] = input[itrial][np.where(sub_ch_list == 3)[0][0], timepoint]
                if 4 in sub_ch_list:
                    mapp[itrial, 1, 1, timepoint] = input[itrial][np.where(sub_ch_list == 4)[0][0], timepoint]
                if 5 in sub_ch_list:
                    mapp[itrial, 1, 3, timepoint] = input[itrial][np.where(sub_ch_list == 5)[0][0], timepoint]
                if 6 in sub_ch_list:
                    mapp[itrial, 1, 5, timepoint] = input[itrial][np.where(sub_ch_list == 6)[0][0], timepoint]
                if 7 in sub_ch_list:
                    mapp[itrial, 1, 7, timepoint] = input[itrial][np.where(sub_ch_list == 7)[0][0], timepoint]
                if 8 in sub_ch_list:
                    mapp[itrial, 1, 9, timepoint] = input[itrial][np.where(sub_ch_list == 8)[0][0], timepoint]
                if 9 in sub_ch_list:
                    mapp[itrial, 2, 0, timepoint] = input[itrial][np.where(sub_ch_list == 9)[0][0], timepoint]
                if 10 in sub_ch_list:
                    mapp[itrial, 2, 2, timepoint] = input[itrial][np.where(sub_ch_list == 10)[0][0], timepoint]
                if 11 in sub_ch_list:
                    mapp[itrial, 2, 4, timepoint] = input[itrial][np.where(sub_ch_list == 11)[0][0], timepoint]
                if 12 in sub_ch_list:
                    mapp[itrial, 2, 6, timepoint] = input[itrial][np.where(sub_ch_list == 12)[0][0], timepoint]
                if 13 in sub_ch_list:
                    mapp[itrial, 2, 8, timepoint] = input[itrial][np.where(sub_ch_list == 13)[0][0], timepoint]
                if 14 in sub_ch_list:
                    mapp[itrial, 2, 10, timepoint] = input[itrial][np.where(sub_ch_list == 14)[0][0], timepoint]
                if 15 in sub_ch_list:
                    mapp[itrial, 3, 1, timepoint] = input[itrial][np.where(sub_ch_list == 15)[0][0], timepoint]
                if 16 in sub_ch_list:
                    mapp[itrial, 3, 3, timepoint] = input[itrial][np.where(sub_ch_list == 16)[0][0], timepoint]
                if 17 in sub_ch_list:
                    mapp[itrial, 3, 5, timepoint] = input[itrial][np.where(sub_ch_list == 17)[0][0], timepoint]
                if 18 in sub_ch_list:
                    mapp[itrial, 3, 7, timepoint] = input[itrial][np.where(sub_ch_list == 18)[0][0], timepoint]
                if 19 in sub_ch_list:
                    mapp[itrial, 3, 9, timepoint] = input[itrial][np.where(sub_ch_list == 19)[0][0], timepoint]
                if 20 in sub_ch_list:
                    mapp[itrial, 4, 2, timepoint] = input[itrial][np.where(sub_ch_list == 20)[0][0], timepoint]
                if 21 in sub_ch_list:
                    mapp[itrial, 4, 4, timepoint] = input[itrial][np.where(sub_ch_list == 21)[0][0], timepoint]
                if 22 in sub_ch_list:
                    mapp[itrial, 4, 6, timepoint] = input[itrial][np.where(sub_ch_list == 22)[0][0], timepoint]
                if 23 in sub_ch_list:
                    mapp[itrial, 4, 8, timepoint] = input[itrial][np.where(sub_ch_list == 23)[0][0], timepoint]
                if 24 in sub_ch_list:
                    mapp[itrial, 5, 1, timepoint] = input[itrial][np.where(sub_ch_list == 24)[0][0], timepoint]
                if 25 in sub_ch_list:
                    mapp[itrial, 5, 3, timepoint] = input[itrial][np.where(sub_ch_list == 25)[0][0], timepoint]
                if 26 in sub_ch_list:
                    mapp[itrial, 5, 5, timepoint] = input[itrial][np.where(sub_ch_list == 26)[0][0], timepoint]
                if 27 in sub_ch_list:
                    mapp[itrial, 5, 7, timepoint] = input[itrial][np.where(sub_ch_list == 27)[0][0], timepoint]
                if 28 in sub_ch_list:
                    mapp[itrial, 5, 9, timepoint] = input[itrial][np.where(sub_ch_list == 28)[0][0], timepoint]
                if 29 in sub_ch_list:
                    mapp[itrial, 6, 3, timepoint] = input[itrial][np.where(sub_ch_list == 29)[0][0], timepoint]
                if 30 in sub_ch_list:
                    mapp[itrial, 6, 5, timepoint] = input[itrial][np.where(sub_ch_list == 30)[0][0], timepoint]
                if 31 in sub_ch_list:
                    mapp[itrial, 6, 7, timepoint] = input[itrial][np.where(sub_ch_list == 31)[0][0], timepoint]
    return mapp

for isub in range(30, 60):
    print(isub)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub + 1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

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

    total_data = np.concatenate((tar_data, nontar_data))
    ## standardScaler 해줘보자
    scalers = {}
    for i in range(total_data.shape[2]):
        scalers[i] = StandardScaler()
        total_data[:, :, i] = scalers[i].fit_transform(total_data[:, :, i])

    tar_data = total_data[:50, :, :]
    nontar_data = total_data[50:, :, :]

    tar_data_mapping = convert_to_2d_doorlock_light(isub, tar_data)
    ntar_data_mapping = convert_to_2d_doorlock_light(isub, nontar_data)

    nnnn = 'E:/[9] 졸업논문/2D_MAP/sub' + str(isub) + '2D_Mapped_tar'
    np.savez(nnnn, tar_data_mapping)
    nnnnn = 'E:/[9] 졸업논문/2D_MAP/sub' + str(isub) + '2D_Mapped_ntar'
    np.savez(nnnnn, ntar_data_mapping)

for isub in range(14):
    print(isub)
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub + 1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    for i in range(ntrain):
        target = data['ERP'][:, 150:, data['target'][i][0] - 1, i]
        tar_data.append(target)
        tar_label.append(1)

        for j in range(6):
            if j == (data['target'][i][0] - 1):
                continue
            else:
                nontar_data.append(data['ERP'][:, 150:, j, i])
                nontar_label.append(0)

    total_data = np.concatenate((tar_data, nontar_data))
    ## standardScaler 해줘보자
    scalers = {}
    for i in range(total_data.shape[2]):
        scalers[i] = StandardScaler()
        total_data[:, :, i] = scalers[i].fit_transform(total_data[:, :, i])

    tar_data = total_data[:50, :, :]
    nontar_data = total_data[50:, :, :]

    tar_data_mapping = convert_to_2d_bs(isub, tar_data)
    ntar_data_mapping = convert_to_2d_bs(isub, nontar_data)

    nnnn = 'E:/[9] 졸업논문/2D_MAP/sub' + str(isub) + 'BS_2D_Mapped_tar'
    np.savez(nnnn, tar_data_mapping)
    nnnnn = 'E:/[9] 졸업논문/2D_MAP/sub' + str(isub) + 'BS_2D_Mapped_ntar'
    np.savez(nnnnn, ntar_data_mapping)