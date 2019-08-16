import cv2
import pandas as pd
import re

## training image read
# img_path = 'E:/Manually_Annotated_file_lists/training.csv'
# img_list = pd.read_csv(img_path)
# pattern = re.compile(r'/')
# i = 3
# img_name = re.split(pattern, img_list['subDirectory_filePath'][i])[1]
# image_path = 'E:/Manually_Annotated_Images/' + re.split(pattern, img_list['subDirectory_filePath'][i])[0] + '/' + img_name
image_path = 'test.jpg'

path = 'C:/Users/jhpark/PycharmProjects/test/venv/Lib/site-packages/cv2/data/'
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path+'haarcascade_eye.xml')

# img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('image title', img)
#
# cv2.waitKey(0) & 0xFF
#
# cv2.destroyAllWindows()

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.23, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()