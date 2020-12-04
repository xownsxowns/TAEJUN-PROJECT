import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from scipy import io
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
import keras.backend as K

for repeat_num in range(1,2):
    for isub in range(30,60):
        model_name = 'E:/[9] 졸업논문/model/feature_extraction/CNN(full ch)/model_CNN2_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model = load_model(model_name)

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

        tar_data = np.reshape(tar_data, (ntrain, nlen, nch))
        nontar_data = np.reshape(nontar_data, ((ntrain * 3), nlen, nch))

        train_vali_data = np.concatenate((tar_data, nontar_data))
        train_vali_label = np.concatenate((tar_label, nontar_label))

        ## standardScaler for SVM
        scalers_SVM = {}
        for i in range(train_vali_data.shape[1]):
            scalers_SVM[i] = StandardScaler()
            train_vali_data[:, i, :] = scalers_SVM[i].fit_transform(train_vali_data[:, i, :])

        new_train_data = train_vali_data.reshape((train_vali_data.shape[0], (train_vali_data.shape[1] * train_vali_data.shape[2])))
        clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
        clf.fit(new_train_data, train_vali_label)

        train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.10,
                                                                          random_state=42)
        ## standardScaler for CNN
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub + 1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        total_label = list()
        total_class = list()
        for i in range(ntest):
            test = data2['ERP'][:, 150:, :, i]
            for j in range(4):
                test_data = test[:, :, j]
                test_data = np.reshape(test_data, (1, nlen, nch))

                test_data_svm = test[:, :, j]
                test_data_svm = np.reshape(test_data_svm, (1, nlen, nch))

                for k in range(test_data.shape[1]):
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])

                for aa in range(test_data_svm.shape[1]):
                    test_data_svm[:, aa, :] = scalers_SVM[aa].transform(test_data_svm[:, aa, :])

                new_test_data = test_data_svm.reshape((test_data_svm.shape[0], (test_data_svm.shape[1] * test_data_svm.shape[2])))
                svm_predicted_class = clf.predict(new_test_data)

                test_data = np.expand_dims(test_data, axis=1)
                predicted_class = model.predict_classes(test_data)

                if svm_predicted_class[0] == 1:
                    total_class.append(svm_predicted_class[0])
                elif predicted_class[0][0] == 0:
                    total_class.append(predicted_class[0][0])
                else:
                    total_class.append(svm_predicted_class[0])

                if j == (data2['target'][i][0]-1):
                    total_label.append(1)
                else:
                    total_label.append(0)

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/' \
                   'P300_Result_SVM_CNN_full_ch_t' + str(repeat_num) + '_confusion_' + str(isub+1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model

    for isub in range(14):
        model_name = 'E:/[9] 졸업논문/model/feature_extraction/CNN(full ch)/model_BS_CNN2_t' + str(repeat_num) + '_train' + str(isub + 1) + '.h5'
        model = load_model(model_name)

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

        tar_data = np.reshape(tar_data, (ntrain, nlen, nch))
        nontar_data = np.reshape(nontar_data, ((ntrain * 5), nlen, nch))

        train_vali_data = np.concatenate((tar_data, nontar_data))
        train_vali_label = np.concatenate((tar_label, nontar_label))

        ## standardScaler for svm
        scalers_SVM = {}
        for i in range(train_vali_data.shape[1]):
            scalers_SVM[i] = StandardScaler()
            train_vali_data[:, i, :] = scalers_SVM[i].fit_transform(train_vali_data[:, i, :])

        new_train_data = train_vali_data.reshape((train_vali_data.shape[0], (train_vali_data.shape[1] * train_vali_data.shape[2])))
        clf = SVC(probability=True, kernel='sigmoid', gamma='auto_deprecated')
        clf.fit(new_train_data, train_vali_label)

        train_data, vali_data, train_label, vali_label = train_test_split(train_vali_data, train_vali_label, test_size=0.10,
                                                                          random_state=42)
        ## standardScaler 해줘보자
        scalers = {}
        for i in range(train_data.shape[1]):
            scalers[i] = StandardScaler()
            train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
            vali_data[:, i, :] = scalers[i].transform(vali_data[:, i, :])

        ## Test
        path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch_BS/Sub' + str(isub + 1) + '_EP_test.mat'
        data2 = io.loadmat(path)
        corr_ans = 0
        ntest = np.shape(data2['ERP'])[3]

        total_label = list()
        total_class = list()
        for i in range(ntest):
            test = data2['ERP'][:, 150:, :, i]
            for j in range(6):
                test_data = test[:, :, j]
                test_data = np.reshape(test_data, (1, nlen, nch))

                test_data_svm = test[:, :, j]
                test_data_svm = np.reshape(test_data_svm, (1, nlen, nch))

                for k in range(test_data.shape[1]):
                    test_data[:, k, :] = scalers[k].transform(test_data[:, k, :])

                for aa in range(test_data_svm.shape[1]):
                    test_data_svm[:, aa, :] = scalers_SVM[aa].transform(test_data_svm[:, aa, :])

                new_test_data = test_data_svm.reshape(
                    (test_data_svm.shape[0], (test_data_svm.shape[1] * test_data_svm.shape[2])))
                svm_predicted_class = clf.predict(new_test_data)

                test_data = np.expand_dims(test_data, axis=1)
                predicted_class = model.predict_classes(test_data)

                if svm_predicted_class[0] == 1:
                    total_class.append(svm_predicted_class[0])
                elif predicted_class[0][0] == 0:
                    total_class.append(predicted_class[0][0])
                else:
                    total_class.append(svm_predicted_class[0])

                if j == (data2['target'][i][0]-1):
                    total_label.append(1)
                else:
                    total_label.append(0)

        confusion_mat = confusion_matrix(total_label, total_class)
        df = pd.DataFrame(confusion_mat)
        filename = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/' \
                   'P300_Result_BS_SVM_CNN_full_ch_t' + str(repeat_num) + '_confusion_' + str(isub+1) + '.csv'
        df.to_csv(filename)

        K.clear_session()
        gc.collect()
        del model
