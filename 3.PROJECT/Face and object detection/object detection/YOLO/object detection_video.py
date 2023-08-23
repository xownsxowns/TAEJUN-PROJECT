import cv2
import time
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(classes), 3))

# loading image
cap = cv2.VideoCapture(0) # 0 for webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read() #
    height, width, channels = frame.shape
    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320,320), (0,0,0), True, crop=False) # reduce 416 to 320

    net.setInput(blob)
    outs = net.forward(outputlayers)

    # showing info on screen/ get