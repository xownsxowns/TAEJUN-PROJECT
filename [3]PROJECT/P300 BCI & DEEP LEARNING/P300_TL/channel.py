from scipy import io
path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/chlist_bs.mat'
data = io.loadmat(path)

isub = 0
list_bs = data['chlist_bs'][0][isub][0]

# 공통인수 뽑고, 각 sub별 채널리스트에서 index받아와서 데이터 가져와
bs_ch_list = []
for isub in range(len(data['chlist_bs'][0])):
    bs_ch_list.append(data['chlist_bs'][0][isub][0])

# 공통인수뽑기
elements_in_all = list(set.intersection(*map(set, bs_ch_list)))
set_elements = set(elements_in_all)
# sub별 index 뽑기
totalsub_index = []
for isub in range(len(data['chlist_bs'][0])):
    list_bs = data['chlist_bs'][0][isub][0]
    list_index = [i for i,e in enumerate(list_bs) if e in set_elements]
    totalsub_index.append(list_index)