import mne
## 흠흠 엠엔이

data = mne.io.read_raw_edf('sub1_1_filtered.edf')
visual = mne.viz.plot_raw(data)


print(visual)
