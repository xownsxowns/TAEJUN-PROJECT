import numpy as np
import _pickle as cPickle
from scipy import signal
from sklearn.svm import SVC
import csv
# data: 40x40x8064 [video/trial x channel(~32:EEG) x data]
# channel: Geneva format
# labels: 40x4 [video/trial x label(valence,arousal,dominance,liking)]
# Downsampled: 128Hz, Bandpass: 4.0~45.0Hz, Data length: 63s (3 second pre-trial baseline)
# theta:4-8, alpha:8-12, beta:12-30, gamma:30-

# path
lab_path = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s'
mac_path = '/Volumes/TAEJUN_USB/DEAP/data_preprocessed_python/data_preprocessed_python/s'

# label & data load
path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/EEG FER/'
read_emotion_label = np.load(path + 'emotion_label.npy', allow_pickle=True).item()

# initialize parameter
acc = list()
predict = np.ones((18,40))
passsub = [3,5,11,14]

for itrial in range(1,23):
    if itrial in passsub:
        continue
    test_data = np.array([])
    train_data = np.array([])
    for isub in range(1,23):
        if isub == itrial:
            filepath = lab_path + str(isub).zfill(2) + '.dat'
            x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
            test_data = x['data']
            test_label = read_emotion_label[isub]
            print(isub)
        elif isub not in passsub:
            filepath = lab_path + str(isub).zfill(2) + '.dat'
            x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
            if train_data.size == 0:
                train_data = x['data']
                label = read_emotion_label[isub]
            else:
                train_data = np.concatenate((train_data, x['data']))
                label = np.concatenate((label, read_emotion_label[isub]))
            print(isub)
    ## feature extraction
    # remove baseline, except EEG signal
    # channel: 32, length: 7680
    train_data = train_data[:,:32,384:]
    test_data  = test_data[:,:32,384:]

    train_data = np.transpose(train_data, (0,2,1))
    test_data  = np.transpose(test_data, (0,2,1))

    # for psd
    fs = 128
    N  = np.shape(train_data)[1]

    theta_data = np.ones((train_data.shape[0],train_data.shape[2]))
    alpha_data = np.ones((train_data.shape[0], train_data.shape[2]))
    beta_data = np.ones((train_data.shape[0], train_data.shape[2]))
    gamma_data = np.ones((train_data.shape[0], train_data.shape[2]))

    for ii in range(train_data.shape[0]):
        for ich in range(train_data.shape[2]):
            f, pxx_den = signal.welch(train_data[ii,:,ich], fs=fs, nperseg=1024)
            theta = np.where(np.logical_and(f>=4, f<8))[0]
            alpha = np.where(np.logical_and(f>=8, f<12))[0]
            beta  = np.where(np.logical_and(f>=12, f<30))[0]
            gamma = np.where(np.logical_and(f>=30, f<45))[0]

            theta_psd = np.mean(pxx_den[theta])
            alpha_psd = np.mean(pxx_den[alpha])
            beta_psd  = np.mean(pxx_den[beta])
            gamma_psd = np.mean(pxx_den[gamma])

            theta_data[ii,ich] = theta_psd
            alpha_data[ii,ich] = alpha_psd
            beta_data[ii,ich]  = beta_psd
            gamma_data[ii,ich] = gamma_psd

    psd_feature = np.concatenate((theta_data, alpha_data, beta_data, gamma_data), axis=1)
    svm_clf = SVC()
    svm_clf.fit(psd_feature, label)

    correct_ans = 0

    theta_data = np.ones((test_data.shape[0],test_data.shape[2]))
    alpha_data = np.ones((test_data.shape[0], test_data.shape[2]))
    beta_data = np.ones((test_data.shape[0], test_data.shape[2]))
    gamma_data = np.ones((test_data.shape[0], test_data.shape[2]))

    for ii in range(test_data.shape[0]):
        for ich in range(test_data.shape[2]):
            f, pxx_den = signal.welch(test_data[ii,:,ich], fs=fs, nperseg=1024)
            theta = np.where(np.logical_and(f>=4, f<8))[0]
            alpha = np.where(np.logical_and(f>=8, f<12))[0]
            beta  = np.where(np.logical_and(f>=12, f<30))[0]
            gamma = np.where(np.logical_and(f>=30, f<45))[0]

            theta_psd = np.mean(pxx_den[theta])
            alpha_psd = np.mean(pxx_den[alpha])
            beta_psd  = np.mean(pxx_den[beta])
            gamma_psd = np.mean(pxx_den[gamma])

            theta_data[ii,ich] = theta_psd
            alpha_data[ii,ich] = alpha_psd
            beta_data[ii,ich]  = beta_psd
            gamma_data[ii,ich] = gamma_psd

    psd_test_feature = np.concatenate((theta_data, alpha_data, beta_data, gamma_data), axis=1)

    for i in range(psd_test_feature.shape[0]):
        predicted_label = svm_clf.predict(test_data)
        if predicted_label == test_label[i]:
            correct_ans += 1
        predict[itrial, i] = predicted_label

    acc[itrial] = correct_ans / len(test_data)
    print(acc)

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/EEG FER/'

f = open(path + 'ACC_SVM.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(acc)
f.close()

np.savetxt(path + 'predicted_label_svm.csv', predict, fmt="%d", delimiter=",")




