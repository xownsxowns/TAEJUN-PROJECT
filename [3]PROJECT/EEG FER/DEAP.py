import numpy as np
import _pickle as cPickle

# data: 40x40x8064 [video/trial x channel(~32:EEG) x data]
# channel: Geneva format
# labels: 40x4 [video/trial x label(valence,arousal,dominance,liking)]
# Downsampled: 128Hz, Bandpass: 4.0~45.0Hz, Data length: 63s (3 second pre-trial baseline)
filepath = '//192.168.1.181/office/[08] 공용데이터베이스/EEG/DEAP/data_preprocessed_python/s01.dat'
x = cPickle.load(open(filepath, 'rb'), encoding='latin1')

