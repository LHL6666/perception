#! /usr/bin/python3
# -*- coding: utf-8 -*
import cv2
import rospy
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from my_roborts_camera.msg import my_msg
from cv_bridge import CvBridge, CvBridgeError


# def callback(data):
#     global Cam_Pub, bridge
#     cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
#     Image_Size = (cv_img.shape[:2])[::-1]
#     map1, map2 = cv2.initUndistortRectifyMap(self_parameter, distortion, np.eye(3), self_parameter, Image_Size,
#                                              cv2.CV_16SC2)
#     image_after = cv2.remap(cv_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     Cam_Pub.publish(bridge.cv2_to_imgmsg(image_after, "bgr8"))
#     cv2.imshow("frame", image_after)
#     cv2.waitKey(1)
#
#
# def my_camera():
#     rospy.init_node('Image_after_node', anonymous=True)
#     global Cam_Pub, bridge
#     bridge = CvBridge()
#     rospy.Subscriber('Camera/image_raw', Image, callback)
#     Cam_Pub = rospy.Publisher('/image_after', Image, queue_size=2)
#     rospy.spin()


class image_after:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("Camera/image_raw", Image,self.subscriber_callback)
        self.image_pub = rospy.Publisher('Camera/image_after', my_msg, queue_size=2)
        self.image0 = np.zeros((720, 1280, 3), dtype=np.uint8)  # 初始图像

    def subscriber_callback(self, data):
        try:
            self.image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_size = self.image0.shape
            image = my_msg()
            image.height = image_size[0]
            image.width = image_size[1]
            image.channels = image_size[2]
            image.data = data.data
            self.image_pub.publish(image)
            print('publishing camera after frame')
            # cv2.imshow("Image after", self.image0)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('image_after_node', anonymous=True)
    my_image_after = image_after()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
