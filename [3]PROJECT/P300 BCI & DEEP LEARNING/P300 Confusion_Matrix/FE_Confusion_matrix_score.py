import numpy as np
import pandas as pd

def calculate_precision(confusion_matrix):
    FP = confusion_matrix['1'][0]
    TP = confusion_matrix['1'][1]
    precision = TP / (TP+FP)
    return precision

def calculate_recall(confusion_matrix):
    FN = confusion_matrix['0'][1]
    TP = confusion_matrix['1'][1]
    recall = TP / (TP+FN)
    return recall

def calculate_f1_score(confusion_matrix):
    FP = confusion_matrix['1'][0]
    FN = confusion_matrix['0'][1]
    TP = confusion_matrix['1'][1]
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = 2 * ((precision*recall) / (precision+recall))
    return f1_score

path = 'C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/P300 BCI & DEEP LEARNING/P300_FeatureExtraction/CONFUSION/'
############ DOORLOCK ###############################
#### RAW ####
total_sum_raw_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_raw_doorlock = total_sum_raw_doorlock + result

#### Morph. ####
total_sum_morph_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_RAW_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_morph_doorlock = total_sum_morph_doorlock + result

#### PCA ####
total_sum_pca_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_PCA_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_pca_doorlock = total_sum_pca_doorlock + result

#### CNN ####
total_sum_cnn_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_CNN_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_doorlock = total_sum_cnn_doorlock + result

#### CNN Full ch ####
total_sum_cnn_full_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_CNN_full_ch_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_full_doorlock = total_sum_cnn_full_doorlock + result

#### CNN 2D ####
total_sum_cnn2d_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_CNN_2D_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn2d_doorlock = total_sum_cnn2d_doorlock + result

#### CNN 2D Full ch ####
total_sum_cnn2d_full_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_CNN_2D_fullch_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn2d_full_doorlock = total_sum_cnn2d_full_doorlock + result

#### DCNN ####
total_sum_dcnn_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_DCNN_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_dcnn_doorlock = total_sum_dcnn_doorlock + result

#### ConvLSTM ####
total_sum_convlstm_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_ConvLSTM_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_convlstm_doorlock = total_sum_convlstm_doorlock + result

#### SAE ####
total_sum_sae_doorlock = np.zeros((2,2))
for isub in range(31,46):
    file = path + 'P300_Result_SAE_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_sae_doorlock = total_sum_sae_doorlock + result

############ LAMP ###############################
#### RAW ####
total_sum_raw_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_raw_lamp = total_sum_raw_lamp + result

#### Morph. ####
total_sum_morph_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_RAW_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_morph_lamp = total_sum_morph_lamp + result

#### PCA ####
total_sum_pca_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_PCA_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_pca_lamp = total_sum_pca_lamp + result

#### CNN ####
total_sum_cnn_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_CNN_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_lamp = total_sum_cnn_lamp + result

#### CNN Full ch ####
total_sum_cnn_full_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_CNN_full_ch_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_full_lamp = total_sum_cnn_full_lamp + result

#### CNN 2D ####
total_sum_cnn2d_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_CNN_2D_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn2d_lamp = total_sum_cnn2d_lamp + result

#### CNN 2D Full ch ####
total_sum_cnn2d_full_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_CNN_2D_fullch_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn2d_full_lamp = total_sum_cnn2d_full_lamp + result

#### DCNN ####
total_sum_dcnn_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_DCNN_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_dcnn_lamp = total_sum_dcnn_lamp + result

#### ConvLSTM ####
total_sum_convlstm_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_ConvLSTM_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_convlstm_lamp = total_sum_convlstm_lamp + result

#### SAE ####
total_sum_sae_lamp = np.zeros((2, 2))
for isub in range(46, 61):
    file = path + 'P300_Result_SAE_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_sae_lamp = total_sum_sae_lamp + result

############ SPEAKER ###############################
#### RAW ####
total_sum_raw_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_SVM_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_raw_bs = total_sum_raw_bs + result

#### Morph. ####
total_sum_morph_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_RAW_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_morph_bs = total_sum_morph_bs + result

#### PCA ####
total_sum_pca_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_PCA_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_pca_bs = total_sum_pca_bs + result

#### CNN ####
total_sum_cnn_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_CNN_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_bs = total_sum_cnn_bs + result

#### CNN Full ch ####
total_sum_cnn_full_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_CNN_full_ch_t1_confusion_' + str(isub) + '.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn_full_bs = total_sum_cnn_full_bs + result

#### CNN 2D ####
total_sum_cnn2d_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_CNN_2D_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn2d_bs = total_sum_cnn2d_bs + result

#### CNN 2D Full ch ####
total_sum_cnn2d_full_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_CNN_2D_fullch_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_cnn2d_full_bs = total_sum_cnn2d_full_bs + result

#### DCNN ####
total_sum_dcnn_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_DCNN_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_dcnn_bs = total_sum_dcnn_bs + result

#### ConvLSTM ####
total_sum_convlstm_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_ConvLSTM_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_convlstm_bs = total_sum_convlstm_bs + result

#### SAE ####
total_sum_sae_bs = np.zeros((2, 2))
for isub in range(1, 15):
    file = path + 'P300_Result_BS_SAE_confusion_' + str(isub) + '_t1.csv'
    result = pd.read_csv(file, header=0, index_col=0)
    total_sum_sae_bs = total_sum_sae_bs + result

############ TOTAL ###############################
total_sum_raw = total_sum_raw_doorlock + total_sum_raw_lamp + total_sum_raw_bs
total_sum_morph = total_sum_morph_doorlock + total_sum_morph_lamp + total_sum_morph_bs
total_sum_pca = total_sum_pca_doorlock + total_sum_pca_lamp + total_sum_pca_bs
total_sum_cnn = total_sum_cnn_doorlock + total_sum_cnn_lamp + total_sum_cnn_bs
total_sum_cnn_full = total_sum_cnn_full_doorlock + total_sum_cnn_full_lamp + total_sum_cnn_full_bs
total_sum_cnn2d = total_sum_cnn2d_doorlock + total_sum_cnn2d_lamp + total_sum_cnn2d_bs
total_sum_cnn2d_full = total_sum_cnn2d_full_doorlock + total_sum_cnn2d_full_lamp + total_sum_cnn2d_full_bs
total_sum_dcnn = total_sum_dcnn_doorlock + total_sum_dcnn_lamp + total_sum_dcnn_bs
total_sum_convlstm = total_sum_convlstm_doorlock + total_sum_convlstm_lamp + total_sum_convlstm_bs
total_sum_sae = total_sum_sae_doorlock + total_sum_sae_lamp + total_sum_sae_bs


################ calculating #####################
## door lock
RAW_DL_precision = calculate_precision(total_sum_raw_doorlock)
RAW_DL_recall = calculate_recall(total_sum_raw_doorlock)
RAW_DL_F1 = calculate_f1_score(total_sum_raw_doorlock)
Morp_DL_precision = calculate_precision(total_sum_morph_doorlock)
Morp_DL_recall = calculate_recall(total_sum_morph_doorlock)
Morp_DL_F1 = calculate_f1_score(total_sum_morph_doorlock)
PCA_DL_precision = calculate_precision(total_sum_pca_doorlock)
PCA_DL_recall = calculate_recall(total_sum_pca_doorlock)
PCA_DL_F1 = calculate_f1_score(total_sum_pca_doorlock)
CNN_DL_precision = calculate_precision(total_sum_cnn_doorlock)
CNN_DL_recall = calculate_recall(total_sum_cnn_doorlock)
CNN_DL_F1 = calculate_f1_score(total_sum_cnn_doorlock)
CNN_full_DL_precision = calculate_precision(total_sum_cnn_full_doorlock)
CNN_full_DL_recall = calculate_recall(total_sum_cnn_full_doorlock)
CNN_full_DL_F1 = calculate_f1_score(total_sum_cnn_full_doorlock)
CNN2D_DL_precision = calculate_precision(total_sum_cnn2d_doorlock)
CNN2D_DL_recall = calculate_recall(total_sum_cnn2d_doorlock)
CNN2D_DL_F1 = calculate_f1_score(total_sum_cnn2d_doorlock)
CNN2D_full_DL_precision = calculate_precision(total_sum_cnn2d_full_doorlock)
CNN2D_full_DL_recall = calculate_recall(total_sum_cnn2d_full_doorlock)
CNN2D_full_DL_F1 = calculate_f1_score(total_sum_cnn2d_full_doorlock)
DCNN_DL_precision = calculate_precision(total_sum_dcnn_doorlock)
DCNN_DL_recall = calculate_recall(total_sum_dcnn_doorlock)
DCNN_DL_F1 = calculate_f1_score(total_sum_dcnn_doorlock)
ConvLSTM_DL_precision = calculate_precision(total_sum_convlstm_doorlock)
ConvLSTM_DL_recall = calculate_recall(total_sum_convlstm_doorlock)
ConvLSTM_DL_F1 = calculate_f1_score(total_sum_convlstm_doorlock)
SAE_DL_precision = calculate_precision(total_sum_sae_doorlock)
SAE_DL_recall = calculate_recall(total_sum_sae_doorlock)
SAE_DL_F1 = calculate_f1_score(total_sum_sae_doorlock)

#### Electric light #####
RAW_EL_precision = calculate_precision(total_sum_raw_lamp)
RAW_EL_recall = calculate_recall(total_sum_raw_lamp)
RAW_EL_F1 = calculate_f1_score(total_sum_raw_lamp)
Morp_EL_precision = calculate_precision(total_sum_morph_lamp)
Morp_EL_recall = calculate_recall(total_sum_morph_lamp)
Morp_EL_F1 = calculate_f1_score(total_sum_morph_lamp)
PCA_EL_precision = calculate_precision(total_sum_pca_lamp)
PCA_EL_recall = calculate_recall(total_sum_pca_lamp)
PCA_EL_F1 = calculate_f1_score(total_sum_pca_lamp)
CNN_EL_precision = calculate_precision(total_sum_cnn_lamp)
CNN_EL_recall = calculate_recall(total_sum_cnn_lamp)
CNN_EL_F1 = calculate_f1_score(total_sum_cnn_lamp)
CNN_full_EL_precision = calculate_precision(total_sum_cnn_full_lamp)
CNN_full_EL_recall = calculate_recall(total_sum_cnn_full_lamp)
CNN_full_EL_F1 = calculate_f1_score(total_sum_cnn_full_lamp)
CNN2D_EL_precision = calculate_precision(total_sum_cnn2d_lamp)
CNN2D_EL_recall = calculate_recall(total_sum_cnn2d_lamp)
CNN2D_EL_F1 = calculate_f1_score(total_sum_cnn2d_lamp)
CNN2D_full_EL_precision = calculate_precision(total_sum_cnn2d_full_lamp)
CNN2D_full_EL_recall = calculate_recall(total_sum_cnn2d_full_lamp)
CNN2D_full_EL_F1 = calculate_f1_score(total_sum_cnn2d_full_lamp)
DCNN_EL_precision = calculate_precision(total_sum_dcnn_lamp)
DCNN_EL_recall = calculate_recall(total_sum_dcnn_lamp)
DCNN_EL_F1 = calculate_f1_score(total_sum_dcnn_lamp)
ConvLSTM_EL_precision = calculate_precision(total_sum_convlstm_lamp)
ConvLSTM_EL_recall = calculate_recall(total_sum_convlstm_lamp)
ConvLSTM_EL_F1 = calculate_f1_score(total_sum_convlstm_lamp)
SAE_EL_precision = calculate_precision(total_sum_sae_lamp)
SAE_EL_recall = calculate_recall(total_sum_sae_lamp)
SAE_EL_F1 = calculate_f1_score(total_sum_sae_lamp)

##### Bluetooth speaker
RAW_BS_precision = calculate_precision(total_sum_raw_bs)
RAW_BS_recall = calculate_recall(total_sum_raw_bs)
RAW_BS_F1 = calculate_f1_score(total_sum_raw_bs)
Morp_BS_precision = calculate_precision(total_sum_morph_bs)
Morp_BS_recall = calculate_recall(total_sum_morph_bs)
Morp_BS_F1 = calculate_f1_score(total_sum_morph_bs)
PCA_BS_precision = calculate_precision(total_sum_pca_bs)
PCA_BS_recall = calculate_recall(total_sum_pca_bs)
PCA_BS_F1 = calculate_f1_score(total_sum_pca_bs)
CNN_BS_precision = calculate_precision(total_sum_cnn_bs)
CNN_BS_recall = calculate_recall(total_sum_cnn_bs)
CNN_BS_F1 = calculate_f1_score(total_sum_cnn_bs)
CNN_full_BS_precision = calculate_precision(total_sum_cnn_full_bs)
CNN_full_BS_recall = calculate_recall(total_sum_cnn_full_bs)
CNN_full_BS_F1 = calculate_f1_score(total_sum_cnn_full_bs)
CNN2D_BS_precision = calculate_precision(total_sum_cnn2d_bs)
CNN2D_BS_recall = calculate_recall(total_sum_cnn2d_bs)
CNN2D_BS_F1 = calculate_f1_score(total_sum_cnn2d_bs)
CNN2D_full_BS_precision = calculate_precision(total_sum_cnn2d_full_bs)
CNN2D_full_BS_recall = calculate_recall(total_sum_cnn2d_full_bs)
CNN2D_full_BS_F1 = calculate_f1_score(total_sum_cnn2d_full_bs)
DCNN_BS_precision = calculate_precision(total_sum_dcnn_bs)
DCNN_BS_recall = calculate_recall(total_sum_dcnn_bs)
DCNN_BS_F1 = calculate_f1_score(total_sum_dcnn_bs)
ConvLSTM_BS_precision = calculate_precision(total_sum_convlstm_bs)
ConvLSTM_BS_recall = calculate_recall(total_sum_convlstm_bs)
ConvLSTM_BS_F1 = calculate_f1_score(total_sum_convlstm_bs)
SAE_BS_precision = calculate_precision(total_sum_sae_bs)
SAE_BS_recall = calculate_recall(total_sum_sae_bs)
SAE_BS_F1 = calculate_f1_score(total_sum_sae_bs)

RAW_precision = calculate_precision(total_sum_raw)
RAW_recall = calculate_recall(total_sum_raw)
RAW_F1 = calculate_f1_score(total_sum_raw)
Morp_precision = calculate_precision(total_sum_morph)
Morp_recall = calculate_recall(total_sum_morph)
Morp_F1 = calculate_f1_score(total_sum_morph)
PCA_precision = calculate_precision(total_sum_pca)
PCA_recall = calculate_recall(total_sum_pca)
PCA_F1 = calculate_f1_score(total_sum_pca)
CNN_precision = calculate_precision(total_sum_cnn)
CNN_recall = calculate_recall(total_sum_cnn)
CNN_F1 = calculate_f1_score(total_sum_cnn)
CNN_full_precision = calculate_precision(total_sum_cnn_full)
CNN_full_recall = calculate_recall(total_sum_cnn_full)
CNN_full_F1 = calculate_f1_score(total_sum_cnn_full)
CNN2D_precision = calculate_precision(total_sum_cnn2d)
CNN2D_recall = calculate_recall(total_sum_cnn2d)
CNN2D_F1 = calculate_f1_score(total_sum_cnn2d)
CNN2D_full_precision = calculate_precision(total_sum_cnn2d_full)
CNN2D_full_recall = calculate_recall(total_sum_cnn2d_full)
CNN2D_full_F1 = calculate_f1_score(total_sum_cnn2d_full)
DCNN_precision = calculate_precision(total_sum_dcnn)
DCNN_recall = calculate_recall(total_sum_dcnn)
DCNN_F1 = calculate_f1_score(total_sum_dcnn)
ConvLSTM_precision = calculate_precision(total_sum_convlstm)
ConvLSTM_recall = calculate_recall(total_sum_convlstm)
ConvLSTM_F1 = calculate_f1_score(total_sum_convlstm)
SAE_precision = calculate_precision(total_sum_sae)
SAE_recall = calculate_recall(total_sum_sae)
SAE_F1 = calculate_f1_score(total_sum_sae)

