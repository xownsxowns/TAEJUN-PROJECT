import cv2
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class_name = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
path = 'C:/Users/xowns/anaconda3/envs/TensorFlow/Lib/site-packages/cv2/data/'
faceCascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')
model = load_model('C:/Users/xowns/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/Face and object detection/Face detection & emotion recognition/model2.h5')


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.23,
        minNeighbors=5,
        minSize=(30, 30)
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        ## roi를 뽑아내는 코드 (roi_color shape??)
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (64,64))
        roi_color = roi_color.astype('float') / 255.0
        roi_color = np.expand_dims(roi_color, axis = 0)
        preds = model.predict(roi_color)[0]
        percen = (preds[preds.argmax()]/sum(preds)) * 100
        percen = "%0.2f" % percen
        label = class_name[preds.argmax()]
        print(label, percen)
        # label display
        y_re = y - 15 if y - 15 > 15 else y + 15
        label_percen = label+':'+str(percen)+'%'
        cv2.putText(frame, str(label_percen), (x, y_re), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0))
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()