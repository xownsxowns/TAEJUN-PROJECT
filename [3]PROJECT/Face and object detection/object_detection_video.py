# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

classNames = eval(open('C:/Users/jhpark/Documents/GitHub/Python_project/[3]PROJECT/Face and object detection/labels.txt').read())



def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    #                              0.007843, (300, 300), 127.5)
    blob = cv2.dnn.blobFromImage(frame,size=(300,300), swapRB=True)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > .6:
            # class_id = detection[1]
            class_id = int(detections[0,0,i,1])
            class_name = id_class_name(class_id, classNames)

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(class_name, confidence * 100)
            print("[INFO] {}".format(label))

            cv2.rectangle(frame, (startX, startY), (endX, endY), (23, 230, 210),
                          thickness=2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, class_name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0))

    # show the output frame
    cv2.imshow("Frame", frame)
    ke = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if ke == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()