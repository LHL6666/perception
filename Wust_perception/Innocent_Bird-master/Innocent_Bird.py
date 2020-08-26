#  CopyRight @LHL 2020.08.13
#  Based on yolov5 model, PyTorch
#  Tested on mi pro , i7-8550U@1.8GHz, MX150-2G, RAM 8.0G
#  Inference FPS:40 ~ 100, auto exposure, inference time: 0.01 ~ 0.024 s

import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import numpy as np
import cv2

cv2.namedWindow("Result",  cv2.WINDOW_NORMAL)
cv2.namedWindow("Original",  cv2.WINDOW_NORMAL)

cv2.moveWindow("Original",   0, 0)
cv2.resizeWindow("Result", 740, 900)

cv2.moveWindow("Result",   450, 10)
cv2.resizeWindow("Original", 1280, 720)

# mainly parameter
########################################################################
FPS = 0.0

# car position
car_x = 0
car_y = 0

# the size field, see in the readme
field_x = 254
field_y0 = 340
field_y1 = 354

# Camera source
Use_Camera = 1
# Screenshots Flag
Shot_Flag = False
# show img
show_image = True
# tracking flag
track_flag = False

################################################################################
# The field area
left_top = (704, 76)
right_top = (940, 103)
left_top_list = [left_top[0], left_top[1]]
Right_top_list = [right_top[0], right_top[1]]

left_bottom = (51, 525)
right_bottom = (812, 716)
left_bottom_list = [left_bottom[0], left_bottom[1]]
right_bottom_list = [right_bottom[0], right_bottom[1]]

List_pst = [left_top_list, Right_top_list, left_bottom_list, right_bottom_list]

################################################################################

# camera parameter
self_parameter = np.array([[736.61, 0., 653.54],
                           [0., 738.53, 364.79],
                           [0., 0., 1.]])

distortion = np.array([[-0.3820, 0.1374, 0., 0., 0.]])


# When running on windows,add the cv2.CAP_DSHOW param
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the frame size
capture.set(3, 1280)
capture.set(4,  720)

# read the first frame
_, img = capture.read()


# Get the image shape
Image_Size = (img.shape[:2])[::-1]
Image_W = Image_Size[0]
Image_H = Image_Size[1]


# parameter for transform
pst1 = np.float32(List_pst)
pst2 = np.float32([[0, 0], [Image_W, 0], [0, Image_H], [Image_W, Image_H]])

# Calculate the transform matrix
M = cv2.getPerspectiveTransform(pst1, pst2)
A = np.array([[List_pst[0][0], List_pst[0][1]]], dtype='float32')
A = np.array([A])

# Set the size to inference
model_size = (256, 256)

# set the model path and select cpu/gpu
weights = 'Jetson_exp12/weights/Innocent_Bird.pt'
device = torch_utils.select_device()

# Load model  and  check image size
model = attempt_load(weights, map_location=device)
image_size = check_img_size(model_size[0], s=model.stride.max())
model.half()

# speed up constant image size inference
cudnn.benchmark = True

# Get names and rectangle color
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


# resize input image for preparing inference
def letterbox(img, new_shape=(256, 256), color=(114, 114, 114), auto=True, scale_up=True):
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
def detect(Bird_img):
    global car_x
    global car_y
    global field_x
    global field_y0
    global field_y1

    boxed_image = letterbox(Bird_img, model_size)
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
        global track_flag
        Loss_Detection = False

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image_data.shape[2:], det[:, :4], Bird_img.shape).round()

            # Print results
            for c in det[:, -1].detach().unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            # Write results
            for *xy, conf, cls in det:
                if show_image:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)

                    labels_list.append(label)

                    if label.startswith('red_car'):
                        if conf <= 0.55:
                            Loss_Detection = True
                        color = (0, 0, 255)
                        # ret = tracker.init(Bird_img, (pst1[0], pst1[1], pst2[0], pst2[1]))
                    elif label.startswith('blue_car'):
                        if conf <= 0.55:
                            Loss_Detection = True
                        color = (255, 0, 0)
                        # ret = tracker.init(Bird_img, (pst1[0], pst1[1], pst2[0], pst2[1]))
                    elif label.startswith('armor'):
                        color = (255, 204, 66)
                    elif label.startswith('tail'):
                        color = (9, 125, 255)
                    else:
                        color = colors[int(cls)]

                    # draw rect and put text
                    if label.startswith('red_car') or label.startswith('blue_car'):
                        pst1, pst2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                        Car_Center = ((pst1[0]+pst2[0])/2, (pst1[1]+pst2[1])/2)

                        # *****************************mainly***************************
                        # 场地半宽x=254cm, y0=340cm, y1=354cm  ，得出的car_x, car_y为该种坐标系下实际的1:1坐标
                        if ref_flag == 0:
                            # adjust_r为调整系数，field_y1为y1(逆透视图中图像底部到参考点对应的实际场地垂直距离), field_y0即指y0(逆透视图中由参考点到图像顶部对应的实际场地垂直距离)
                            adjust_r = field_y1 / field_y0
                            # Car_Center[1]指逆透视图中机器人在height方向上的位置，Car_Center[0]指逆透视图中机器人在width方向上的位置 
                            if Car_Center[1] < ref_point[1]:
                                # y0那边的区域
                                car_y = ((ref_point[1] - Car_Center[1]) / ref_point[1]) * field_y0 * adjust_r
                                car_x = ((Car_Center[0] - ref_point[0]) / Bird_img.shape[1]) * field_x * adjust_r
                            elif Car_Center[1] > ref_point[1]:
                                # y1这边的区域
                                car_x = ((Car_Center[0] - ref_point[0]) / Bird_img.shape[1]) * field_x * adjust_r
                                car_y = ((Car_Center[1] - ref_point[1]) / (Bird_img.shape[0] - ref_point[1])) * (-field_y1) * adjust_r

                            print("car position : ", "x=%d" % car_x, "y=%d" % car_y)
                            cv2.putText(Bird_img, "x= %d y=%d" % (car_x, car_y), ((int)(Car_Center[0]), (int)(Car_Center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                        if conf > 0.55:
                            plot_one_box(xy, Bird_img, label=label, color=color, line_thickness=2)
                        else:
                            pass
                    else:
                        plot_one_box(xy, Bird_img, label=label, color=color, line_thickness=2)


# about reference point
ref_flag = 1
ref_point = (0, 0)
ref_point_t = (0, 0)


# for calibration, print the coordinate
def mouse_event(event, x, y, flags, param):
    global ref_flag
    global ref_point
    global ref_point_t
    if event == cv2.EVENT_LBUTTONDOWN:
        if ref_flag == 1:
            ref_flag = 2
            ref_point_t = (x, y)
            print("Point:", ref_point_t)
        else:
            temp_point = (x, y)
            # print the point clicked
            print("Point", temp_point)
            # Double click mouse to set the reference point
            if abs(ref_point_t[0]-temp_point[0] < 3) and abs(ref_point_t[1]-temp_point[1] < 3) and ref_flag == 2:
                ref_point = (round((ref_point_t[0]+temp_point[0])/2), round((ref_point_t[1]+temp_point[1])/2))
                print("Reference P:", ref_point)
                ref_flag = 0


# for getting train samples
count = 0
number = 1

# *************Prediction motivation****************
# tracker_types = ['MOSSE', 'CSRT']
# tracker_type = tracker_types[0]
#
# # create tracker
# if tracker_type == 'MOSSE':
#     tracker = cv2.TrackerMOSSE_create()
# elif tracker_type == 'CSRT':
#     tracker = cv2.TrackerCSRT_create()

while True:
    ret, frame = capture.read()
    cv2.setMouseCallback("Result", mouse_event)

    # Correction of my camera distortion
    map1, map2 = cv2.initUndistortRectifyMap(self_parameter, distortion, np.eye(3), self_parameter, Image_Size, cv2.CV_16SC2)
    image = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # to indicate roi area and referent point(field center)
    cv2.circle(image, left_top, 2, (0, 255, 0), thickness=2)
    cv2.circle(image, right_top, 2, (0, 255, 0), thickness=2)
    cv2.circle(image, ref_point,  2, (0, 99, 99), thickness=2)
    cv2.circle(image, left_bottom, 2, (0, 255, 0), thickness=2)
    cv2.circle(image, right_bottom, 2, (0, 255, 0), thickness=2)

    cv2.rectangle(image, (left_top[0]+4, left_top[1]+4), (left_top[0]-4, left_top[1]-4), (0, 0, 255), thickness=2)
    cv2.rectangle(image, (right_top[0]+4, right_top[1]+4), (right_top[0]-4, right_top[1]-4), (0, 0, 255), thickness=2)
    cv2.rectangle(image, (ref_point[0] + 4, ref_point[1]+4), (ref_point[0]-4, ref_point[1]-4), (0, 255, 0), thickness=2)
    cv2.rectangle(image, (left_bottom[0]+4,   left_bottom[1]+4), (left_bottom[0]-4,   left_bottom[1]-4), (0, 0, 255), thickness=2)
    cv2.rectangle(image, (right_bottom[0] + 4,  right_bottom[1]+4), (right_bottom[0]-4, right_bottom[1]-4), (0, 0, 255), thickness=2)

    Bird_img = cv2.warpPerspective(image, M, (Image_W, Image_H))
    Bird_img = cv2.resize(Bird_img, (720, 900))

    # detection
    detect(Bird_img)

    cv2.imshow("Result", Bird_img)

    # show axis
    cv2.arrowedLine(image, (530, 150), (704, 10), (0, 128, 255), 2, 0, 0, 0.1)
    cv2.arrowedLine(image, (530, 150), (1280, 205), (0, 255, 120), 2, 0, 0, 0.1)
    cv2.imshow("Original", image)

    if cv2.waitKey(1) == ord('s') and not Shot_Flag:
        Shot_Flag = True
        print("Start to screenshots!")
    elif cv2.waitKey(1) == ord('q'):
        capture.release()
        break

    # after press 's', to collect training images
    if Shot_Flag:
        count +=1
        if count >= 70:
            count = 0
            number += 1
            Success = cv2.imwrite("E:/image/car_img/capture_image/Bird%s.jpg" % number, Bird_img)
            if (Success):
                print("Successfully saved bird%s image" % number)
            else:
                print("Failed to save image")

cv2.destroyAllWindows()
