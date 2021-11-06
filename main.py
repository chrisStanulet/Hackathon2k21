import cv2
import imutils
from feed import LiveCamera

LBDet = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
#UPDet = cv2.CascadeClassifier('haarcascade_upperbody.xml')
#FBDet = cv2.CascadeClassifier('haarcascade_fullbody.xml')

a = LiveCamera()
a.start()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):
    color, depth = a.get()
    cv2.imshow("Image",color)

    color = imutils.resize(color,
                           width=min(400, color.shape[1]))

    #color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    LB = LBDet.detectMultiScale(color,1.1,4)
    #UB = UPDet.detectMultiScale(color,1.05,5)
    #FB = FBDet.detectMultiScale(color,1.1,4)

    (regions, _) = hog.detectMultiScale(color,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.1)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(color, (x, y),
                      (x + w, y + h),
                      (255, 0, 0), 2)
        print(depth[x + int(w/2)][y + int(h/2)])


    # Drawing the regions in the Image
    for (x, y, w, h) in LB:
        cv2.rectangle(color, (x, y),
                      (x + w, y + h),
                      (0, 0, 0), 2)

    # Showing the output Image
    cv2.imshow("Image", color)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

a.stop()
cv2.destroyAllWindows()
