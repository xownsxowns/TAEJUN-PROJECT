
## P300 Classification
## CNN feature extraction

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

from scipy import io, signal
import pandas as pd
import numpy as np
import random
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from imblearn.under_sampling import *
from sklearn.metrics import accuracy_score
import gc
import keras.backend as K
from sklearn.metrics import confusion_matrix

# parameter setting

np.random.seed(0)
random.seed(0)

def convert_to_2d_doorlock_light(sub_num, input):
    ch_path1 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/ch/chlist_doorlock'
    ch_path2 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/ch/chlist_light'
    ch_data1 = io.loadmat(ch_path1)
    ch_data2 = io.loadmat(ch_path2)
    ch_list  = []
    for i in range(len(ch_data1['chlist_doorlock'][0])):
        ch_list.append(ch_data1['chlist_doorlock'][0][i][0])
    for ii in range(len(ch_data2['chlist_light'][0])):
        ch_list.append(ch_data2['chlist_light'][0][ii][0])
    sub_ch_list = ch_list[sub_num-30]
    mapp = np.zeros((np.shape(input)[0],7,11,np.shape(input)[2]))
    for itrial in range(np.shape(input)[0]):
        for timepoint in range(np.shape(input)[2]):
            for a in range(np.shape(input)[1]):
                if 1 in sub_ch_list:
                    mapp[itrial, 0, 3, timepoint] = input[itrial][np.where(sub_ch_list==1)[0][0], timepoint]
                if 2 in sub_ch_list:
                    mapp[itrial, 0, 5, timepoint] = input[itrial][np.where(sub_ch_list==2)[0][0], timepoint]
                if 3 in sub_ch_list:
                    mapp[itrial, 0, 7, timepoint] = input[itrial][np.where(sub_ch_list==3)[0][0], timepoint]
                if 4 in sub_ch_list:
                    mapp[itrial, 1, 1, timepoint] = input[itrial][np.where(sub_ch_list==4)[0][0], timepoint]
                if 5 in sub_ch_list:
                    mapp[itrial, 1, 3, timepoint] = input[itrial][np.where(sub_ch_list==5)[0][0], timepoint]
                if 6 in sub_ch_list:
                    mapp[itrial, 1, 5, timepoint] = input[itrial][np.where(sub_ch_list==6)[0][0], timepoint]
                if 7 in sub_ch_list:
                    mapp[itrial, 1, 7, timepoint] = input[itrial][np.where(sub_ch_list==7)[0][0], timepoint]
                if 8 in sub_ch_list:
                    mapp[itrial, 1, 9, timepoint] = input[itrial][np.where(sub_ch_list==8)[0][0], timepoint]
                if 9 in sub_ch_list:
                    mapp[itrial, 2, 0, timepoint] = input[itrial][np.where(sub_ch_list==9)[0][0], timepoint]
                if 10 in sub_ch_list:
                    mapp[itrial, 2, 2, timepoint] = input[itrial][np.where(sub_ch_list==10)[0][0], timepoint]
                if 11 in sub_ch_list:
                    mapp[itrial, 2, 4, timepoint] = input[itrial][np.where(sub_ch_list==11)[0][0], timepoint]
                if 12 in sub_ch_list:
                    mapp[itrial, 2, 6, timepoint] = input[itrial][np.where(sub_ch_list==12)[0][0], timepoint]
                if 13 in sub_ch_list:
                    mapp[itrial, 2, 8, timepoint] = input[itrial][np.where(sub_ch_list==13)[0][0], timepoint]
                if 14 in sub_ch_list:
                    mapp[itrial, 2, 10, timepoint] = input[itrial][np.where(sub_ch_list==14)[0][0], timepoint]
                if 15 in sub_ch_list:
                    mapp[itrial, 3, 1, timepoint] = input[itrial][np.where(sub_ch_list==15)[0][0], timepoint]
                if 16 in sub_ch_list:
                    mapp[itrial, 3, 3, timepoint] = input[itrial][np.where(sub_ch_list==16)[0][0], timepoint]
                if 17 in sub_ch_list:
                    mapp[itrial, 3, 5, timepoint] = input[itrial][np.where(sub_ch_list==17)[0][0], timepoint]
                if 18 in sub_ch_list:
                    mapp[itrial, 3, 7, timepoint] = input[itrial][np.where(sub_ch_list==18)[0][0], timepoint]
                if 19 in sub_ch_list:
                    mapp[itrial, 3, 9, timepoint] = input[itrial][np.where(sub_ch_list==19)[0][0], timepoint]
                if 20 in sub_ch_list:
                    mapp[itrial, 4, 2, timepoint] = input[itrial][np.where(sub_ch_list==20)[0][0], timepoint]
                if 21 in sub_ch_list:
                    mapp[itrial, 4, 4, timepoint] = input[itrial][np.where(sub_ch_list==21)[0][0], timepoint]
                if 22 in sub_ch_list:
                    mapp[itrial, 4, 6, timepoint] = input[itrial][np.where(sub_ch_list==22)[0][0], timepoint]
                if 23 in sub_ch_list:
                    mapp[itrial, 4, 8, timepoint] = input[itrial][np.where(sub_ch_list==23)[0][0], timepoint]
                if 24 in sub_ch_list:
                    mapp[itrial, 5, 1, timepoint] = input[itrial][np.where(sub_ch_list==24)[0][0], timepoint]
                if 25 in sub_ch_list:
                    mapp[itrial, 5, 3, timepoint] = input[itrial][np.where(sub_ch_list==25)[0][0], timepoint]
                if 26 in sub_ch_list:
                    mapp[itrial, 5, 5, timepoint] = input[itrial][np.where(sub_ch_list==26)[0][0], timepoint]
                if 27 in sub_ch_list:
                    mapp[itrial, 5, 7, timepoint] = input[itrial][np.where(sub_ch_list==27)[0][0], timepoint]
                if 28 in sub_ch_list:
                    mapp[itrial, 5, 9, timepoint] = input[itrial][np.where(sub_ch_list==28)[0][0], timepoint]
                if 29 in sub_ch_list:
                    mapp[itrial, 6, 3, timepoint] = input[itrial][np.where(sub_ch_list==29)[0][0], timepoint]
                if 30 in sub_ch_list:
                    mapp[itrial, 6, 5, timepoint] = input[itrial][np.where(sub_ch_list==30)[0][0], timepoint]
                if 31 in sub_ch_list:
                    mapp[itrial, 6, 7, timepoint] = input[itrial][np.where(sub_ch_list==31)[0][0], timepoint]
    return mapp

def convert_to_2d_bs(sub_num, input):
    ch_path1 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/ch/chlist_bs'
    ch_data1 = io.loadmat(ch_path1)
    ch_list  = []
    for i in range(len(ch_data1['chlist_bs'][0])):
        ch_list.append(ch_data1['chlist_bs'][0][i][0])
    sub_ch_list = ch_list[sub_num]
    mapp = np.zeros((np.shape(input)[0],7,11,np.shape(input)[2]))
    for itrial in range(np.shape(input)[0]):
        for timepoint in range(np.shape(input)[2]):
            for a in range(np.shape(input)[1]):
                if 1 in sub_ch_list:
                    mapp[itrial, 0, 3, timepoint] = input[itrial][np.where(sub_ch_list==1)[0][0], timepoint]
                if 2 in sub_ch_list:
                    mapp[itrial, 0, 5, timepoint] = input[itrial][np.where(sub_ch_list==2)[0][0], timepoint]
                if 3 in sub_ch_list:
                    mapp[itrial, 0, 7, timepoint] = input[itrial][np.where(sub_ch_list==3)[0][0], timepoint]
                if 4 in sub_ch_list:
                    mapp[itrial, 1, 1, timepoint] = input[itrial][np.where(sub_ch_list==4)[0][0], timepoint]
                if 5 in sub_ch_list:
                    mapp[itrial, 1, 3, timepoint] = input[itrial][np.where(sub_ch_list==5)[0][0], timepoint]
                if 6 in sub_ch_list:
                    mapp[itrial, 1, 5, timepoint] = input[itrial][np.where(sub_ch_list==6)[0][0], timepoint]
                if 7 in sub_ch_list:
                    mapp[itrial, 1, 7, timepoint] = input[itrial][np.where(sub_ch_list==7)[0][0], timepoint]
                if 8 in sub_ch_list:
                    mapp[itrial, 1, 9, timepoint] = input[itrial][np.where(sub_ch_list==8)[0][0], timepoint]
                if 9 in sub_ch_list:
                    mapp[itrial, 2, 0, timepoint] = input[itrial][np.where(sub_ch_list==9)[0][0], timepoint]
                if 10 in sub_ch_list:
                    mapp[itrial, 2, 2, timepoint] = input[itrial][np.where(sub_ch_list==10)[0][0], timepoint]
                if 11 in sub_ch_list:
                    mapp[itrial, 2, 4, timepoint] = input[itrial][np.where(sub_ch_list==11)[0][0], timepoint]
                if 12 in sub_ch_list:
                    mapp[itrial, 2, 6, timepoint] = input[itrial][np.where(sub_ch_list==12)[0][0], timepoint]
                if 13 in sub_ch_list:
                    mapp[itrial, 2, 8, timepoint] = input[itrial][np.where(sub_ch_list==13)[0][0], timepoint]
                if 14 in sub_ch_list:
                    mapp[itrial, 2, 10, timepoint] = input[itrial][np.where(sub_ch_list==14)[0][0], timepoint]
                if 15 in sub_ch_list:
                    mapp[itrial, 3, 1, timepoint] = input[itrial][np.where(sub_ch_list==15)[0][0], timepoint]
                if 16 in sub_ch_list:
                    mapp[itrial, 3, 3, timepoint] = input[itrial][np.where(sub_ch_list==16)[0][0], timepoint]
                if 17 in sub_ch_list:
                    mapp[itrial, 3, 5, timepoint] = input[itrial][np.where(sub_ch_list==17)[0][0], timepoint]
                if 18 in sub_ch_list:
                    mapp[itrial, 3, 7, timepoint] = input[itrial][np.where(sub_ch_list==18)[0][0], timepoint]
                if 19 in sub_ch_list:
                    mapp[itrial, 3, 9, timepoint] = input[itrial][np.where(sub_ch_list==19)[0][0], timepoint]
                if 20 in sub_ch_list:
                    mapp[itrial, 4, 2, timepoint] = input[itrial][np.where(sub_ch_list==20)[0][0], timepoint]
                if 21 in sub_ch_list:
                    mapp[itrial, 4, 4, timepoint] = input[itrial][np.where(sub_ch_list==21)[0][0], timepoint]
                if 22 in sub_ch_list:
                    mapp[itrial, 4, 6, timepoint] = input[itrial][np.where(sub_ch_list==22)[0][0], timepoint]
                if 23 in sub_ch_list:
                    mapp[itrial, 4, 8, timepoint] = input[itrial][np.where(sub_ch_list==23)[0][0], timepoint]
                if 24 in sub_ch_list:
                    mapp[itrial, 5, 1, timepoint] = input[itrial][np.where(sub_ch_list==24)[0][0], timepoint]
                if 25 in sub_ch_list:
                    mapp[itrial, 5, 3, timepoint] = input[itrial][np.where(sub_ch_list==25)[0][0], timepoint]
                if 26 in sub_ch_list:
                    mapp[itrial, 5, 5, timepoint] = input[itrial][np.where(sub_ch_list==26)[0][0], timepoint]
                if 27 in sub_ch_list:
                    mapp[itrial, 5, 7, timepoint] = input[itrial][np.where(sub_ch_list==27)[0][0], timepoint]
                if 28 in sub_ch_list:
                    mapp[itrial, 5, 9, timepoint] = input[itrial][np.where(sub_ch_list==28)[0][0], timepoint]
                if 29 in sub_ch_list:
                    mapp[itrial, 6, 3, timepoint] = input[itrial][np.where(sub_ch_list==29)[0][0], timepoint]
                if 30 in sub_ch_list:
                    mapp[itrial, 6, 5, timepoint] = input[itrial][np.where(sub_ch_list==30)[0][0], timepoint]
                if 31 in sub_ch_list:
                    mapp[itrial, 6, 7, timepoint] = input[itrial][np.where(sub_ch_list==31)[0][0], timepoint]
    return mapp


for repeat_num in range(1,2):
    total_acc = list()
    train_score = list()
    for isub in range(30,60):
        adasyn = NeighbourhoodCleaningRule()
        model_name = 'E:/[9] 졸업논문/model/undersampling/NCR/CNN2D Full ch/model_CNN2D_fullch_ncr_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model = load_model(model_name)

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

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        total_label = list()
        total_class = list()

        for i in range(ntest):
            test = data2['ERP'][:,150:,:,i]
            total_prob = list()
            for j in range(4):
                test_data = test[:, :, j]
                test_data = np.expand_dims(test_data, axis=0)
                for k in range(test_data.shape[2]):
                    test_data[:, :, k] = scalers[k].transform(test_data[:, :, k])
                test_data_mapping = convert_to_2d_doorlock_light(isub, test_data)
                test_data_mapping = np.transpose(test_data_mapping, (0, 3, 1, 2))
                test_data = np.expand_dims(test_data_mapping, axis=1)
                predicted_class = model.predict_classes(test_data)
                total_class.append(predicted_class[0][0])
                if j == (data2['target'][i][0]-1):
                    total_label.append(1)
                else:
                    total_label.append(0)

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/' \
                   'CNN2D_fullch_ncr_t' + str(repeat_num) + '_confusion_' + str(isub+1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model

    for isub in range(14):
        adasyn = NeighbourhoodCleaningRule()
        model_name = 'E:/[9] 졸업논문/model/undersampling/NCR/CNN2D Full ch/model_BS_CNN2D_fullch_ncr_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model = load_model(model_name)

        print(isub)
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_training.mat'
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

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub+1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        total_label = list()
        total_class = list()
        for i in range(ntest):
            test = data2['ERP'][:,150:,:,i]
            total_prob = list()
            for j in range(6):
                test_data = test[:, :, j]
                test_data = np.expand_dims(test_data, axis=0)
                for k in range(test_data.shape[2]):
                    test_data[:, :, k] = scalers[k].transform(test_data[:, :, k])
                test_data_mapping = convert_to_2d_bs(isub, test_data)
                test_data_mapping = np.transpose(test_data_mapping, (0, 3, 1, 2))
                test_data = np.expand_dims(test_data_mapping, axis=1)
                predicted_class = model.predict_classes(test_data)
                total_class.append(predicted_class[0][0])
                if j == (data2['target'][i][0] - 1):
                    total_label.append(1)
                else:
                    total_label.append(0)

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_sampling/UnderSampling/CONFUSION/NCR/' \
                    'CNN2D_fullch_BS_ncr_t' + str(repeat_num) + '_confusion_' + str(isub + 1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model

