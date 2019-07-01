import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from scipy import io

for isub in range(60):
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntrain = np.shape(data['ERP'])[3]

    tar_data = list()
    tar_label = list()
    nontar_data = list()
    nontar_label = list()

    # 100ms~600ms 길이 자른것
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

    for ich in range(tar_data.shape[2]):
        # Transform the time series into Gramian Angular Fields
        # image_size -> Shape of output
        gasf = GramianAngularField(image_size=20, method='summation')
        X_gasf = gasf.fit_transform(tar_data[:,:,ich])
        # gadf = GramianAngularField(image_size=10, method='difference')
        # X_gadf = gadf.fit_transform(tar_data[:,:,0])

        # images = [X_gasf[0], X_gadf[0]]
        for itrial in range(np.shape(X_gasf)[0]):
            images = [X_gasf[itrial]]
            titles = ['Gramian Angular Summation Field']

            height = np.shape(images)[1]
            width = np.shape(images)[2]
            figsize = (height, width)
            # Show the images for the first time series
            # fig = plt.figure(figsize=(12, 7))
            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, 1)
                             # axes_pad=0.15,
                             # share_all=True,
                             # cbar_location="right",
                             # cbar_mode="single",
                             # cbar_size="7%",
                             # cbar_pad=0.3,
                             )

            for image, title, ax in zip(images, titles, grid):
                im = ax.imshow(image, cmap='rainbow', origin='lower')
                # ax.set_title(title)
            # ax.cax.colorbar(im)
            # ax.cax.toggle_label(True)
            ax.axis('off')
            plt.xticks([]), plt.yticks([])
            plt.tight_layout()
            plt.subplots_adjust(bottom=0, top=1, right=1, left=0)
            file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub+1) + '/tar_trial' + str(itrial+1) + '_GAF_ch' + str(ich+1) + '.png'
            plt.savefig(file_path,
                        bbox_inces='tight',
                        pad_inches=0)
            plt.clf()
            plt.cla()
            plt.close()

    print('sub{0} is ended'.format(isub+1))
#
# for isub in range(60):
#     path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_training.mat'
#     data = io.loadmat(path)
#
#     nch = np.shape(data['ERP'])[0]
#     nlen = 250
#     ntrain = np.shape(data['ERP'])[3]
#
#     tar_data = list()
#     tar_label = list()
#     nontar_data = list()
#     nontar_label = list()
#
#     # 100ms~600ms 길이 자른것
#     for i in range(ntrain):
#         target = data['ERP'][:, 150:, data['target'][i][0] - 1, i]
#         tar_data.append(target)
#         tar_label.append(1)
#
#         for j in range(4):
#             if j == (data['target'][i][0] - 1):
#                 continue
#             else:
#                 nontar_data.append(data['ERP'][:, 150:, j, i])
#                 nontar_label.append(0)
#
#     tar_data = np.reshape(tar_data, (ntrain, nlen, nch))
#     nontar_data = np.reshape(nontar_data, ((ntrain * 3), nlen, nch))
#
#     for ich in range(nontar_data.shape[2]):
#         # Transform the time series into Gramian Angular Fields
#         # image_size -> Shape of output
#         gasf = GramianAngularField(image_size=20, method='summation')
#         X_gasf = gasf.fit_transform(nontar_data[:,:,ich])
#         # gadf = GramianAngularField(image_size=10, method='difference')
#         # X_gadf = gadf.fit_transform(tar_data[:,:,0])
#
#         # images = [X_gasf[0], X_gadf[0]]
#         for itrial in range(np.shape(X_gasf)[0]):
#             images = [X_gasf[itrial]]
#             titles = ['Gramian Angular Summation Field']
#
#             height = np.shape(images)[1]
#             width = np.shape(images)[2]
#             figsize = (height, width)
#             # Show the images for the first time series
#             # fig = plt.figure(figsize=(12, 7))
#             fig = plt.figure(figsize=figsize)
#             grid = ImageGrid(fig, 111,
#                              nrows_ncols=(1, 1)
#                              # axes_pad=0.15,
#                              # share_all=True,
#                              # cbar_location="right",
#                              # cbar_mode="single",
#                              # cbar_size="7%",
#                              # cbar_pad=0.3,
#                              )
#
#             for image, title, ax in zip(images, titles, grid):
#                 im = ax.imshow(image, cmap='rainbow', origin='lower')
#                 # ax.set_title(title)
#             # ax.cax.colorbar(im)
#             # ax.cax.toggle_label(True)
#             ax.axis('off')
#             plt.xticks([]), plt.yticks([])
#             plt.tight_layout()
#             plt.subplots_adjust(bottom=0, top=1, right=1, left=0)
#             file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub+1) + '/nontar_trial' + str(itrial+1) + '_GAF_ch' + str(ich+1) + '.png'
#             plt.savefig(file_path,
#                         bbox_inces='tight',
#                         pad_inches=0)
#             plt.clf()
#             plt.cla()
#             plt.close()
#
#     print('sub{0} is ended'.format(isub+1))
