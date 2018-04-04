import mne


data = mne.io.read_raw_edf('sub1_1_filtered.edf')
visual = mne.viz.plot_raw(data)


print(visual)