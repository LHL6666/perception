# coding:utf-8

import cv2
import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image
from my_roborts_camera.msg import my_msg
from cv_bridge import CvBridge, CvBridgeError

# camera parameter
self_parameter = np.array([[736.61, 0., 653.54],
                           [0., 738.53, 364.79],
                           [0., 0., 1.]])

distortion = np.array([[-0.3820, 0.1374, 0., 0., 0.]])


def image_capture():
    rospy.init_node("Camera_Source_Node", anonymous=True)
    # set queue_size small enough for real_time
    image_publish = rospy.Publisher('Camera/image_raw', Image, queue_size=2)

    rate = rospy.Rate(20)

    capture = cv2.VideoCapture(2)

    # Set the frame size
    capture.set(3, 1280)
    capture.set(4, 720)

    # the 'CVBridge' is a python_class, must have a instance.
    # That means "cv2_to_imgmsg() must be called with CvBridge instance"
    bridge = CvBridge()

    if not capture.isOpened():
        sys.stdout.write("Camera is not available !")
        return -1

    count = 0
    while not rospy.is_shutdown():
        ret, frame = capture.read()
        image_size = (frame.shape[:2])[::-1]
        map1, map2 = cv2.initUndistortRectifyMap(self_parameter, distortion, np.eye(3), self_parameter, image_size,
                                                 cv2.CV_16SC2)
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if ret:
            pass
        else:
            rospy.loginfo("Capturing image failed.")

        # cv2.imshow("image_after", frame)
        # cv2.waitKey(1)
        image1 = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        image_publish.publish(image1)
        print('publishing camera frame')
        rate.sleep()


if __name__ == '__main__':
    image_capture()
