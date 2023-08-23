import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from scipy import io

for isub in range(60):
    ## Test
    path = 'E:/[1] Experiment/[1] BCI/P300LSTM/Epoch_data/Epoch/Sub' + str(isub+1) + '_EP_test.mat'
    data = io.loadmat(path)

    nch = np.shape(data['ERP'])[0]
    nlen = 250
    ntest = np.shape(data['ERP'])[3]
    nstim = 4

    test_data = list()
    # ntest, nch, nlen, nstim
    # 100ms~600ms 길이 자른것
    for i in range(ntest):
        target = data['ERP'][:, 150:, :, i]
        test_data.append(target)
    # nstim, ntest, nlen, nch
    test_data = np.transpose(test_data, (3, 0, 2, 1))

    for nstim in range(test_data.shape[0]):
        for ich in range(test_data.shape[3]):
            # Transform the time series into Gramian Angular Fields
            # image_size -> Shape of output
            gasf = GramianAngularField(image_size=20, method='summation')
            X_gasf = gasf.fit_transform(test_data[nstim,:,:,ich])
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
                file_path = 'E:/[1] Experiment/[1] BCI/P300LSTM/GAFimage/' + 'sub' + str(isub+1) + '/test_trial' + str(nstim+1) + '-' + str(itrial+1) + '_GAF_ch' + str(ich+1) + '.png'
                plt.savefig(file_path,
                            bbox_inces='tight',
                            pad_inches=0)
                plt.clf()
                plt.cla()
                plt.close()

    print('sub{0} is ended'.format(isub + 1))