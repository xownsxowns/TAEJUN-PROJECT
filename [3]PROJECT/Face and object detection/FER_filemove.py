import pandas as pd
import re
import cv2
import numpy as np
import shutil

## training image read
img_path = 'E:/Manually_Annotated_file_lists/training.csv'
img_list = pd.read_csv(img_path)
pattern = re.compile(r'/')
pattern2 = re.compile(r'[/.]')
img_train = list()

label_list = [0,1,2,3,4,5,6]

# for i in range(img_list.shape[0]):
#     if img_list['expression'][i] in label_list:
#         img_name = re.split(pattern, img_list['subDirectory_filePath'][i])[1]
#         image_path = 'E:/Manually_Annotated_Images/' + re.split(pattern, img_list['subDirectory_filePath'][i])[
#             0] + '/' + img_name
#
#         img = cv2.imread(image_path)
#         img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_AREA)
#
#         # scaling into [0, 1]
#         img_tensor = img / 255.
#
#         img_train.append(img_tensor)
#         print(i)
#     else:
#         print(i)
#         continue
#
# data1 = np.asarray(img_train)
# np.savez_compressed('C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/Face and object detection/data1', data1=data1)

## 전체 데이터 npz로 한개씩 저장
# save_path = 'E:/[2] 연구/[3] Facial/npz_data/'
# for i in range(img_list.shape[0]):
#     if img_list['expression'][i] in label_list:
#         img_name = re.split(pattern, img_list['subDirectory_filePath'][i])[1]
#         image_path = 'E:/Manually_Annotated_Images/' + re.split(pattern, img_list['subDirectory_filePath'][i])[
#             0] + '/' + img_name
#
#         img = cv2.imread(image_path)
#         img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_AREA)
#
#         # scaling into [0, 1]
#         img_tensor = img / 255.
#
#         # label
#         # label.append(img_list['expression'][i])
#
#         data = np.asarray(img_tensor)
#
#         data_path = save_path + 'data' + str(num)
#         np.savez_compressed(data_path, data=data)
#         num = num+1
#         print(i)
#     else:
#         print(i)
#         continue

# label_data = np.asarray(label)
# np.savez_compressed([save_path+'label'], label=label_data)

save_path = 'E:/[2] 연구/[3] Facial/npz_data/'
num_neu, num_hap, num_sad, num_sur, num_fear, num_anger, num_dis = 1,1,1,1,1,1,1
for i in range(img_list.shape[0]):
    if img_list['expression'][i] in label_list:
        img_name = re.split(pattern, img_list['subDirectory_filePath'][i])[1]
        image_path = 'E:/Manually_Annotated_Images/' + re.split(pattern, img_list['subDirectory_filePath'][i])[
            0] + '/' + img_name
        name = re.split(pattern2, img_name)
        if img_list['expression'][i] == 0:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Neutral/'
            shutil.copy(image_path,new_path + 'Neutral' + str(num_neu) + '.' + name[1])
            num_neu += 1
        elif img_list['expression'][i] == 1:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Happy/'
            shutil.copy(image_path, new_path + 'Happy' + str(num_hap) + '.' + name[1])
            num_hap += 1
        elif img_list['expression'][i] == 2:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Sad/'
            shutil.copy(image_path, new_path + 'Sad' + str(num_sad) + '.' + name[1])
            num_sad += 1
        elif img_list['expression'][i] == 3:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Surprise/'
            shutil.copy(image_path, new_path + 'Surprise' + str(num_sur) + '.' + name[1])
            num_sur += 1
        elif img_list['expression'][i] == 4:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Fear/'
            shutil.copy(image_path, new_path + 'Fear' + str(num_fear) + '.' + name[1])
            num_fear += 1
        elif img_list['expression'][i] == 5:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Disgust/'
            shutil.copy(image_path, new_path + 'Disgust' + str(num_dis) + '.' + name[1])
            num_dis += 1
        elif img_list['expression'][i] == 6:
            new_path = 'E:/[2] 연구/[3] Facial/training_set/Anger/'
            shutil.copy(image_path, new_path + 'Anger' + str(num_anger) + '.' + name[1])
            num_anger += 1
        print(i)
    else:
        print(i)
        continue
