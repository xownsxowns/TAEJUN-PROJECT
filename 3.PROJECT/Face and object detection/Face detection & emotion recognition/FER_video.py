import cv2
from keras.models import load_model
import numpy as np
import time

np.set_printoptions(precision=2)

# class 0:Anger, 1:Disgust, 2:Fear, 3:Happy, 4:Neutral, 5:Sad, 6:Surprise

class_name = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
path = 'C:/Users/jhpark/PycharmProjects/test/venv/Lib/site-packages/cv2/data/'
faceCascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')
model = load_model('C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/Face and object detection/Face detection & emotion recognition/model2.h5')

# # setting video range
# start_time = 0
# end_time = 6000

# video_path = 'E:/[2] 연구/[3] Facial/test_video.avi'
video_path = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/Face and object detection/Face detection & emotion recognition/s06_trial06.avi'

video_capture = cv2.VideoCapture(video_path)
# video_capture.get(cv2.CAP_PROP_FPS)
# video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time)

score_collect = list()

##### For FPS #####
prevTime = 0 # for saving previous time

while(video_capture.isOpened()):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.23,
            minNeighbors=5,
            minSize=(30, 30)
            # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        ### Get current Time
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1/(sec)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
            ## roi를 뽑아내는 코드 (roi_color shape??)
            roi_color = frame[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (64,64))
            roi_color = roi_color.astype('float') / 255.0
            roi_color = np.expand_dims(roi_color, axis = 0)
            preds = model.predict(roi_color)[0]
            score_collect.append(preds)
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
        # print("FPS:%0.1f" % fps)
        cv2.imshow('Video', frame)

    # if video_capture.get(cv2.CAP_PROP_POS_MSEC) == end_time:
    #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# When everything is done, release the capture
score = np.sum(score_collect, axis=0) / np.size(score_collect, axis=0)
score = (np.array(score)*100).tolist()
print('Anger:{:0.2f}%, Fear:{:0.2f}%, Happy:{:0.2f}%, Neutral:{:0.2f}%, Sad:{:0.2f}%, Surprise:{:0.2f}%'.format(
    score[0],score[1],score[2],score[3],score[4],score[5]))

video_capture.release()
cv2.destroyAllWindows()