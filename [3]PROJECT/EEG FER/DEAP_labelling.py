## Building a model using AffectNet dataset

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# training image read
img_path = 'E:/Manually_Annotated_file_lists/training.csv'
img_list = pd.read_csv(img_path)

# extract target expression
# 0: NEU, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger

data = img_list.loc[(img_list['expression'] == 0) | (img_list['expression'] == 1) | (img_list['expression'] == 2) |
                    (img_list['expression'] == 3) | (img_list['expression'] == 4) | (img_list['expression'] == 5) |
                    (img_list['expression'] == 6)]

feature = data[['valence','arousal']]
label   = data[['expression']]

neigh = KNeighborsClassifier()
neigh.fit(feature, np.ravel(label))

# svm = SVC()
# svm.fit(feature, np.ravel(label))
#
# mlp = MLPClassifier()
# mlp.fit(feature, np.ravel(label))
#
# ada = AdaBoostClassifier()
# ada.fit(feature, np.ravel(label))

tree = DecisionTreeClassifier()
tree.fit(feature, np.ravel(label))

# Test image read
timg_path = 'E:/Manually_Annotated_file_lists/validation.csv'
timg_list = pd.read_csv(timg_path)
data_test = timg_list.loc[(timg_list['expression'] == 0) | (timg_list['expression'] == 1) | (timg_list['expression'] == 2) |
                         (timg_list['expression'] == 3) | (timg_list['expression'] == 4) | (timg_list['expression'] == 5) |
                         (timg_list['expression'] == 6)]
feature_test = data_test[['valence','arousal']]
label_test   = data_test[['expression']]

neigh.score(feature_test, label_test)
# svm.score(feature_test, label_test)
# mlp.score(feature_test, label_test)
# ada.score(feature_test, label_test)
tree.score(feature_test, label_test)

## DEAP dataset labelling
import numpy as np
import _pickle as cPickle

# Labeling
# 0: NEU, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger
NEU_allsub, HAP_allsub, SAD_allsub, SUR_allsub, FEA_allsub, DIS_allsub, ANG_allsub = {},{},{},{},{},{},{}
NEU_label_allsub, HAP_label_allsub, SAD_label_allsub, SUR_label_allsub, FEA_label_allsub, DIS_label_allsub, ANG_label_allsub = {},{},{},{},{},{},{}
emotion_label = {}

# path
lab_path = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s'
mac_path = '/Volumes/TAEJUN_USB/DEAP/data_preprocessed_python/data_preprocessed_python/s'

for isub in range(1,23):
    filepath = lab_path + str(isub).zfill(2) + '.dat'
    x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
    data = x['data']
    label = x['labels']
    NEU_list, HAP_list, SAD_list, SUR_list, FEA_list, DIS_list, ANG_list = [], [], [], [], [], [], []
    discrete_emotion = []

    for i in range(np.size(label, axis=0)):
        deap_feature = [(label[i][0]-5)/4, (label[i][1]-5)/4]
        deap_feature = np.reshape(deap_feature, (1, 2))
        deap_label = tree.predict(deap_feature)[0]
        if deap_label == 0:
            NEU_list.append(i)
            discrete_emotion.append(0)
        elif deap_label == 1:
            HAP_list.append(i)
            discrete_emotion.append(1)
        elif deap_label == 2:
            SAD_list.append(i)
            discrete_emotion.append(2)
        elif deap_label == 3:
            SUR_list.append(i)
            discrete_emotion.append(3)
        elif deap_label == 4:
            FEA_list.append(i)
            discrete_emotion.append(4)
        elif deap_label == 5:
            DIS_list.append(i)
            discrete_emotion.append(5)
        elif deap_label == 6:
            ANG_list.append(i)
            discrete_emotion.append(6)

    NEU, NEU_label  = data[NEU_list,:,:], np.zeros((np.size(NEU_list, axis=0)))
    HAP, HAP_label = data[HAP_list, :, :], np.ones((np.size(HAP_list, axis=0)))
    SAD, SAD_label = data[SAD_list, :, :], np.ones((np.size(SAD_list, axis=0))) * 2
    SUR, SUR_label = data[SUR_list, :, :], np.ones((np.size(SUR_list, axis=0))) * 3
    FEA, FEA_label = data[FEA_list, :, :], np.ones((np.size(FEA_list, axis=0))) * 4
    DIS, DIS_label = data[DIS_list, :, :], np.ones((np.size(DIS_list, axis=0))) * 5
    ANG, ANG_label = data[ANG_list, :, :], np.ones((np.size(ANG_list, axis=0))) * 6

    NEU_allsub[isub], NEU_label_allsub[isub] = NEU, NEU_label
    HAP_allsub[isub], HAP_label_allsub[isub] = HAP, HAP_label
    SAD_allsub[isub], SAD_label_allsub[isub] = SAD, SAD_label
    SUR_allsub[isub], SUR_label_allsub[isub] = SUR, SUR_label
    FEA_allsub[isub], FEA_label_allsub[isub] = FEA, FEA_label
    DIS_allsub[isub], DIS_label_allsub[isub] = DIS, DIS_label
    ANG_allsub[isub], ANG_label_allsub[isub] = ANG, ANG_label

    emotion_label[isub] = discrete_emotion

    print('sub{0} ended'.format(isub))

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/EEG FER/'
np.save(path + 'emotion_label.npy', emotion_label)