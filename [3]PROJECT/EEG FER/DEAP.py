import numpy as np
import _pickle as cPickle

# data: 40x40x8064 [video/trial x channel(~32:EEG) x data]
# channel: Geneva format
# labels: 40x4 [video/trial x label(valence,arousal,dominance,liking)]
# Downsampled: 128Hz, Bandpass: 4.0~45.0Hz, Data length: 63s (3 second pre-trial baseline)
# filepath = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s01.dat'
# x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
# data = x['data']
# label = x['labels']

# 1: positive valence + high arousal
# 2: positive valence + low arousal
# 3: negative valence + high arousal
# 4: negative valence + low arousal

# Labeling
PVHA_allsub, PVLA_allsub, NVHA_allsub, NVLA_allsub, NEU_allsub = {},{},{},{},{}
PVHA_label_allsub, PVLA_label_allsub, NVHA_label_allsub, NVLA_label_allsub, NEU_label_allsub = {},{},{},{},{}

for isub in range(1,23):
    filepath = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s'+ str(isub) + '.dat'
    x = cPickle.load(open(filepath, 'rb'), encoding='latin1')
    data = x['data']
    label = x['labels']
    PVHA_list, PVLA_list, NVHA_list, NVLA_list, NEU_list = [], [], [], [], []

    for i in range(np.size(label, axis=0)):
        if label[i][0] > 4 and label[i][1] > 4:
            PVHA_list.append(i)
        elif label[i][0] > 4 and label[i][1] < 4:
            PVLA_list.append(i)
        elif label[i][0] < 4 and label[i][1] > 4:
            NVHA_list.append(i)
        elif label[i][0] < 4 and label[i][1] < 4:
            NVLA_list.append(i)
        elif label[i][0] < 6 and label[i][0] > 4 and label[i][1] < 6 and label[i][1] > 4:
            NEU_list.append(i)

    PVHA, PVHA_label = data[PVHA_list,:,:], np.ones((np.size(PVHA_list, axis=0)))
    PVLA, PVLA_label = data[PVLA_list,:,:], np.ones((np.size(PVLA_list, axis=0))) * 2
    NVHA, NVHA_label = data[NVHA_list,:,:], np.ones((np.size(NVHA_list, axis=0))) * 3
    NVLA, NVLA_label = data[NVLA_list,:,:], np.ones((np.size(NVLA_list, axis=0))) * 4
    NEU, NEU_label  = data[NEU_list,:,:], np.zeros((np.size(NEU_list, axis=0)))

    PVHA_allsub[isub], PVHA_label_allsub[isub] = PVHA, PVHA_label
    PVLA_allsub[isub], PVLA_label_allsub[isub] = PVLA, PVLA_label
    NVHA_allsub[isub], NVHA_label_allsub[isub] = NVHA, NVHA_label
    NVLA_allsub[isub], NVLA_label_allsub[isub] = NVLA, NVLA_label
    NEU_allsub[isub], NEU_label_allsub[isub] = NEU, NEU_label

    print('sub{0} ended'.format(isub))



