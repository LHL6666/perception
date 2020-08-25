import cv2
import time
import numpy as np

fps = 0.0

cv2.setUseOptimized(onoff=True)

self_parameter = np.array([[736.61, 0., 653.54],
                           [0., 738.53, 364.79],
                           [0., 0., 1.]])

distortion = np.array([[-0.3820, 0.1374, 0., 0., 0.]])

capture=cv2.VideoCapture(1, cv2.CAP_DSHOW)
capture.set(3, 1280)
capture.set(4,  720)
# capture.set(5,   100)
# # capture.set(15,  -2)

while True:
    t1 = time.time()

    ret, frame = capture.read()
    Image_Size = (frame.shape[:2])[::-1]
    map1, map2 = cv2.initUndistortRectifyMap(self_parameter, distortion, np.eye(3), self_parameter, Image_Size, cv2.CV_16SC2)

    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    frame = cv2.medianBlur(frame, 3)
    # frame = cv2.GaussianBlur(frame, (5, 5), 1)
    fps = (fps + (1. / (time.time() - t1))) / 2.0

    frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) == ord('q'):
        capture.release()
        break

cv2.destroyAllWindows()
