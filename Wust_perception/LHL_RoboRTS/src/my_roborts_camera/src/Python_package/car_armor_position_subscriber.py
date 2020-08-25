#! /usr/bin/python2.7
# coding: utf-8

import rospy
from my_roborts_camera.msg import car_armor_position


def publish_info_callback(my_msg):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("All objects: Car: {}   String: {}  Armor: {}".format(my_msg.car, my_msg.str, my_msg.armor))

    print("Cars number:  red_car1: {}   red_car2: {}  blue_car1: {}  blue_car2: {}".format(my_msg.red_car1, my_msg.red_car2, my_msg.blue_car1, my_msg.blue_car2))

    print("Red car position:  left_t_x {}, left_t_y: {}, right_b_x: {}, right_b_y: {}".format(my_msg.red_top_left_x, my_msg.red_top_left_y, my_msg.red_bottom_right_x, my_msg.red_bottom_right_y))

    print("Blue car position:  left_t_x {}, left_t_y: {}, right_b_x: {}, right_b_y: {}".format(my_msg.blue_top_left_x, my_msg.blue_top_left_y, my_msg.blue_bottom_right_x, my_msg.blue_bottom_right_y))

    print("Credible armor position:  left_t_x {}, left_t_y: {}, right_b_x: {}, right_b_y: {}".format(my_msg.armor_top_left_x, my_msg.armor_top_left_y, my_msg.armor_bottom_right_x, my_msg.armor_bottom_right_y))

    print("The temporary target position: x1: {}    y1: {}    x2: {}    y2: {}".format(my_msg.temp_top_left_x, my_msg.temp_top_left_y, my_msg.temp_bottom_right_x, my_msg.temp_bottom_right_x))



def msg_subscription():
    rospy.init_node('Car_Armor_Str_Msg_Node', anonymous=True)
    rospy.Subscriber('MSG/car_armor_position_msg', car_armor_position, publish_info_callback)
    rospy.spin()


def main():
    msg_subscription()
