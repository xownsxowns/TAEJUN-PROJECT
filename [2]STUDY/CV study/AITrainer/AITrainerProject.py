import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Right Arm
        detector.findAnlge(img, 12, 14, 16)
        # Left Arm
        detector.findAngle(img, 11, 13, 15)

    cv2.imshow("Image", img)
    cv2.waitKey(1)