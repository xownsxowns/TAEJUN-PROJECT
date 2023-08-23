import cv2
import sys

# Path = sys.argv[1] if len(sys.argv) > 1 else '.'
path = 'C:/Users/jhpark/PycharmProjects/test/venv/Lib/site-packages/cv2/data/'
faceCascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')

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
        # roi_color = frame[y:y+h, x:x+w]
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()