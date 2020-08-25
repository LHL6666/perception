#! /usr/bin/python3
# -*- coding: utf-8 -*

#  CopyRight @LHL 2020.08.13
#  Based on yolov5 model, PyTorch
#  Tested on mi pro , i7-8550U@1.8GHz, MX150-2G, RAM 8.0G
#  Inference FPS:30 ~ 50, auto exposure, inference time: 0.01 ~ 0.025 s

from my_roborts_camera.msg import car_armor_position
from my_roborts_camera.msg import my_msg
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import numpy as np
import threading
import rospy
import cv2

# red_car armor1, blue_car armor2, select one to continue
enemy = 'armor2'


# resize input image for preparing inference
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scale_up=True):
    # Resize image to a 32*x rectangle
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new_W / old_w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down to get better mAP
    if not scale_up:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # minimum rectangle
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

    # divide padding into 2 sides
    dw /= 2
    dh /= 2
    # to resize image
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


# object detection
def detect_callback(data):
    global enemy, colors, names, device, pre_image, model, model_size, Position_Publish

    Car_Armor_Position_Msgs = car_armor_position()

    armor_num = 0
    red_car_num = 0
    blue_car_num = 0
    red_car_position = []
    blue_car_position = []
    last_armor_conf = 0.5
    last_red_car_conf = 0.5
    last_blue_car_conf = 0.5

    # MSG_Init
    Car_Armor_Position_Msgs.car = 0
    Car_Armor_Position_Msgs.str = 0

    Car_Armor_Position_Msgs.red_car1 = 0
    Car_Armor_Position_Msgs.red_car2 = 0
    Car_Armor_Position_Msgs.blue_car1 = 0
    Car_Armor_Position_Msgs.blue_car2 = 0

    Car_Armor_Position_Msgs.red_top_left_x = 0.0
    Car_Armor_Position_Msgs.red_top_left_y = 0.0
    Car_Armor_Position_Msgs.red_bottom_right_x = 0.0
    Car_Armor_Position_Msgs.red_bottom_right_y = 0.0

    Car_Armor_Position_Msgs.blue_top_left_x = 0.0
    Car_Armor_Position_Msgs.blue_top_left_y = 0.0
    Car_Armor_Position_Msgs.blue_bottom_right_x = 0.0
    Car_Armor_Position_Msgs.blue_bottom_right_y = 0.0

    Car_Armor_Position_Msgs.armor_top_left_x = 0.0
    Car_Armor_Position_Msgs.armor_top_left_y = 0.0
    Car_Armor_Position_Msgs.armor_bottom_right_x = 0.0
    Car_Armor_Position_Msgs.armor_bottom_right_y = 0.0

    Car_Armor_Position_Msgs.temp_top_left_x = 0.0
    Car_Armor_Position_Msgs.temp_top_left_y = 0.0
    Car_Armor_Position_Msgs.temp_bottom_right_x = 0.0
    Car_Armor_Position_Msgs.temp_bottom_right_y = 0.0

    # print("receiving image for starting detection now!")
    image = np.ndarray(shape=(data.height, data.width, data.channels), dtype=np.uint8,
                       buffer=data.data)

    boxed_image = letterbox(image, model_size)
    # Stack
    image_data = np.stack(boxed_image, 0)

    # Convert, BGR to RGB, bsx3x416x416
    image_data = image_data[:, :, ::-1].transpose(2, 0, 1)
    image_data = np.ascontiguousarray(image_data)

    image_data = torch.from_numpy(image_data).to(device)
    # u8 to fp16/32
    image_data = image_data.half()

    # from 0~255 to 0.0~1.0
    image_data /= 255.0
    if image_data.ndimension() == 3:
        image_data = image_data.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()

    predict = model(image_data, augment=False)[0]
    # Apply NMS
    predict = non_max_suppression(predict, 0.4, 0.5, classes=0, agnostic=False)
    t2 = torch_utils.time_synchronized()

    print("Inference Time:", t2 - t1)

    # Process detections
    for i, det in enumerate(predict):
        s = '%g:' % i
        s += '%gx%g ' % image_data.shape[2:]

        labels_list = []
        # print("Image:", image_data.shape[2:])

        if det is not None and len(det):
            # Rescale boxes
            det[:, :4] = scale_coords(image_data.shape[2:], det[:, :4], image.shape).round()

            # Print results
            for c in det[:, -1].detach().unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
                print("Classes:", s)
            # Write results
            for *xy, conf, cls in det:
                if True:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)

                    labels_list.append(label)

                    pst1, pst2 = (xy[0], xy[1]), (xy[2], xy[3])

                    pst1 = np.float32(pst1)
                    pst2 = np.float32(pst2)

                    # if find car
                    if label.startswith('red_car') or label.startswith('blue_car'):
                        if label.startswith('red_car'):
                            red_car_num += 1
                            color = (0, 0, 255)
                            if conf > last_red_car_conf:
                                last_red_car_conf = conf
                                red_car_position.append(pst1)
                                red_car_position.append(pst2)
                                Car_Armor_Position_Msgs.red_top_left_x = pst1[0]
                                Car_Armor_Position_Msgs.red_top_left_y = pst1[1]
                                Car_Armor_Position_Msgs.red_bottom_right_x = pst2[0]
                                Car_Armor_Position_Msgs.red_bottom_right_y = pst2[1]

                            if label.startswith('red_car1'):
                                Car_Armor_Position_Msgs.red_car1 = 1
                            else:
                                Car_Armor_Position_Msgs.red_car2 = 1

                        else:
                            blue_car_num += 1
                            color = (255, 0, 0)
                            if conf > last_blue_car_conf:
                                last_blue_car_conf = conf
                                blue_car_position.append(pst1)
                                blue_car_position.append(pst2)
                                Car_Armor_Position_Msgs.blue_top_left_x = pst1[0]
                                Car_Armor_Position_Msgs.blue_top_left_y = pst1[1]
                                Car_Armor_Position_Msgs.blue_bottom_right_x = pst2[0]
                                Car_Armor_Position_Msgs.blue_bottom_right_y = pst2[1]

                            if label.startswith('blue_car1'):
                                Car_Armor_Position_Msgs.blue_car1 = 1
                            else:
                                Car_Armor_Position_Msgs.blue_car2 = 1
                        Car_Armor_Position_Msgs.car = 1
                    # if find armor
                    elif label.startswith('armor'):
                        # Only record the enemy armor
                        if label.startswith(enemy):
                            armor_num += 1
                            color = (255, 204, 66)
                            Car_Armor_Position_Msgs.armor = 1
                            if conf > last_armor_conf:
                                last_armor_conf = conf
                                Car_Armor_Position_Msgs.armor_top_left_x = pst1[0]
                                Car_Armor_Position_Msgs.armor_top_left_y = pst1[1]
                                Car_Armor_Position_Msgs.armor_bottom_right_x = pst2[0]
                                Car_Armor_Position_Msgs.armor_bottom_right_y = pst2[1]
                        else:
                            color = (0, 255, 0)

                    # if find string or number
                    else:
                        Car_Armor_Position_Msgs.str = 1
                        if label.startswith('Heart'):
                            color = (0, 0, 255)
                            Car_Armor_Position_Msgs.temp_top_left_x = pst1[0]
                            Car_Armor_Position_Msgs.temp_top_left_y = pst1[1]
                            Car_Armor_Position_Msgs.temp_bottom_right_x = pst2[0]
                            Car_Armor_Position_Msgs.temp_bottom_right_y = pst2[1]
                        else:
                            # color = (128, 128, 128)
                            color = colors[int(cls)]

                # draw rect and put text
                if label.startswith('red_car') or label.startswith('blue_car'):
                    if conf > 0.55:
                        plot_one_box(xy, image, label=label, color=color, line_thickness=2)
                    else:
                        plot_one_box(xy, image, label=label, color=(0, 0, 0), line_thickness=2)
                else:
                    plot_one_box(xy, image, label=label, color=color, line_thickness=2)

        cv2.imshow("detection", image)
        cv2.waitKey(1)
        Position_Publish.publish(Car_Armor_Position_Msgs)


def main():
    global colors, names, device, pre_image, model, model_size, Position_Publish
    # **************************
    pre_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    model_size = (512, 448)
    image_size = (720, 1280)
    # **************************

    # set the model path and select cpu/gpu
    weights = '/home/dji/.local/lib/python3.6/site-packages/runs/exp0/weights/best.pt'
    device = torch_utils.select_device()

    # Load model  and  check image size
    model = attempt_load(weights, map_location=device)
    print("load model succeed!")
    image_size = check_img_size(model_size[0], s=model.stride.max())
    model.half()

    # speed up constant image size inference
    cudnn.benchmark = True

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print("start receiving image")
    rospy.init_node('Detection_node', anonymous=True)
    rospy.Subscriber('Camera/image_after', my_msg, detect_callback)
    Position_Publish = rospy.Publisher('MSG/car_armor_position_msg', car_armor_position, queue_size=2)

    detection_thread = threading.Thread(target=rospy.spin)
    detection_thread.start()

