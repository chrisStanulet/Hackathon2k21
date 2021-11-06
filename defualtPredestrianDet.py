import cv2
import imutils
import pyrealsense2 as rs
from feed import LiveCamera
import numpy as np


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

a = LiveCamera()
a.start()
while(True):
    color, depth = a.get()
    color = imutils.resize(color,
                           width=min(400, color.shape[1]))

    (regions, _) = hog.detectMultiScale(color,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(color, (x, y),
                      (x + w, y + h),
                      (0, 0, 255), 2)

    # Showing the output Image
    cv2.imshow("Image", color)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

a.release()
cv2.destroyAllWindows()

