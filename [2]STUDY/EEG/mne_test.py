## MNE
import mne
import scipy import io

# data = mne.io.read_raw_edf('C:/Users/jhpark/Documents/GitHub/Python_project/[2]STUDY/EEG/sub1_1_filtered.edf')
# visual = mne.viz.plot_raw(data)
# mne.viz.plot_topomap(data[24][1])
#
# print(visual)
# print(data)

path = 'E:/[3] 수업/ME특론1/ME1.mat'
data = io.loadmat(path)

# 2 13