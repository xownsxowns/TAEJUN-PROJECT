import cv2

videoFile1 = 'C:/Users/jhpark/Documents/GitHub/TAEJUN PROJECT/[3]PROJECT/Face and object detection/Face detection & emotion recognition/s06_trial06.avi'
cap = cv2.VideoCapture(videoFile1)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        cv2.imshow('video',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()