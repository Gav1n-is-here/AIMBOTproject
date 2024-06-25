

from ultralytics import YOLOv10
# 指定模型权重文件的路径
model_path = ".\models\\v10nbest.pt"
# 加载模型
model = YOLOv10(model_path)
#需要在yolov10依赖基础上安装pip install mss pandas keyboard
import cv2
import numpy as np
import mss

import keyboard
from mouseinput import SendInput ,mouse_input



def stop_loop(e):
    """当'q'键被按下时，改变标志变量的值以停止循环"""
    global keep_running
    keep_running = False
    print("Q键被按下，即将停止循环...")

def toggle_track(e):
    global keep_track
    if keep_track == False:
        keep_track = True
        print("P键被按下继续追踪")
    else:
        keep_track = False
        print("P键被按下暂停追踪")

def toggle_target(e):
    global track_target
    if track_target == False:
        track_target = True
        print("追踪T目标")
    else:
        track_target = False
        print("追踪CT目标")

from ultralytics.engine.results import Boxes
def calculate_center_distance(results_boxes, image_width=640, image_height=640):
    CT=[1,2]
    T=[3,4]
    global track_target
    if track_target == False:
        targetlist=CT
        
    else:
        targetlist=T
    # 确保传入的是Boxes实例
    if isinstance(results_boxes, Boxes):
        # 获取边界框的xyxy格式和类别ID
        boxes_xyxy = results_boxes.xyxy
        classes = results_boxes.cls
        
        # 初始化列表存储类别为1或2的目标中心坐标到图像中心的欧氏距离及其索引
        distances = []
        image_center = (image_width / 2, image_height / 2)
        
        # 遍历所有检测框
        for idx, box in enumerate(boxes_xyxy):
            width = abs(box[2] - box[0])
            height = abs(box[3] - box[1])
            # 判断高度是否大于宽度
            # 检查类别ID是否为1或2
            if classes[idx] in targetlist and height > width:
                # 计算边界框中心坐标
                box_center = (((box[2] + box[0]) / 2), ((box[3] + box[1]) / 2))  # 注意这里的顺序对应xyxy
                
                # 计算该中心点到图像中心的欧氏距离
                distance = np.linalg.norm(np.array(image_center) - np.array(box_center))
                if classes[idx]== targetlist[1]:
                    distance=distance*0.6
                # 存储距离及其索引
                distances.append((distance, box_center))
        
        # 找到距离最小的目标
        if distances:
            min_distance, closest_box_center = min(distances, key=lambda x: x[0])
            # 计算dx和dy
            dx = closest_box_center[0] - image_width / 2
            dy = closest_box_center[1] - image_height / 2
            
            return dx, dy
        else:
            return None, None
    else:
        raise TypeError("Input must be an instance of Boxes class.")


def main():

    with mss.mss() as sct:
        # 自定义截屏区域
        monitor = {"top": 220, "left": 640, "width": 640, "height": 640}
        #调试单次可detect BGR IMG
        # screenshot = sct.grab(monitor)
        # img = np.array(screenshot)
        # # 转换颜色格式从 BGRA 到 BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # results = model(img)
        # results[0].show()
        # # 显示截屏内容
        # cv2.imshow('Screen Capture', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        while keep_running:
            # 截屏
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            # 转换颜色格式从 BGRA 到 BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            results = model(img)
            # results[0].show()
            dx, dy = calculate_center_distance(results[0].boxes)
            if dx is not None and dy is not None:
                dx=int(dx)
                dy=int(dy)
            # print(f"dx: {dx}, dy: {dy}")
                if keep_track == True:
                    SendInput(mouse_input(1, dx, dy))
            # # 显示截屏内容
            # cv2.imshow('Screen Capture', img)

            # # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        # # 释放所有窗口
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    # SendInput(mouse_input(1, 100, 400))测试鼠标移动
    
    # 定义一个标志变量，用于控制循环
    keep_running = True
    keep_track = False
    track_target=True
    # 设置键盘监听，当'q'键被按下时调用stop_loop函数
    keyboard.on_press_key('q', stop_loop)
    keyboard.on_press_key('p', toggle_track)
    keyboard.on_press_key('l', toggle_target)
    main()






