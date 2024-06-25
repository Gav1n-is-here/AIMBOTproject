# python detectbyme.py --weights models/cs088.pt --source "screen 0 640 360 640 360" --project runs/detect --name mss2test 
# python detectbyme.py --weights models/cs088.pt --source "screen 0 1000 400 1240 600" --project runs/detect --name mss2test

import keyboard

import argparse
import csv
import os
import sys
from pathlib import Path
import torch


#鼠标控制
import ctypes
LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort
INPUT_MOUSE = 0
class MouseInput(ctypes.Structure):
    _fields_ = [
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]
class InputUnion(ctypes.Union):
    _fields_ = [
        ('mi', MouseInput)
    ]
class Input(ctypes.Structure):
    _fields_ = [
        ('types', DWORD),
        ('iu', InputUnion)
    ]
def mouse_input_set(flags, x, y, data):
    return MouseInput(x, y, data, flags, 0, None)
def input_do(structure):
    if isinstance(structure, MouseInput):
        return Input(INPUT_MOUSE, InputUnion(mi=structure))
    raise TypeError('Cannot create Input structure!')
def mouse_input(flags, x=0, y=0, data=0):
    return input_do(mouse_input_set(flags, x, y, data))
def SendInput(*inputs):
    n_inputs = len(inputs)
    lp_input = Input * n_inputs
    p_inputs = lp_input(*inputs)
    cb_size = ctypes.c_int(ctypes.sizeof(Input))
    return ctypes.windll.user32.SendInput(n_inputs, p_inputs, cb_size)























FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights,
    source,  # screen
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=10,  # maximum detections per imageq
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images

    # # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if 0 else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    flag =0
    for path, im, im0s, vid_cap, s in dataset:

        target=  headorbody
        team =  TorCT

        if path is None and im is None and im0s is None and vid_cap is None and s is None:
            break
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)


       
        #指定锁定类别
        aimclass=0
        if(team=='T'):
            if(target=='head'):
                aimclass=4
            else:
                aimclass=3
        else:
            if(target=='head'):
                aimclass=2
            else:
                aimclass=1

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += "%gx%g " % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if 0 else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    
                    if save_img :  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if 0 else (names[c] if 0 else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))


                # Process detections
                min_distance = float('inf')  # Initialize with infinity
                dxx=0
                dyy=0
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
    
                   # Filter boxes based on class index (e.g., class 0)
                    if c == aimclass:
                        label = names[c]
                        confidence = float(conf)

                        # Calculate box center coordinates
                        box_center_x = (xyxy[0] + xyxy[2]) / 2
                        box_center_y = (xyxy[1] + xyxy[3]) / 2

                        # Calculate distance from box center to image center
                        dx = box_center_x - im0.shape[1] / 2  # x coordinate
                        dy = box_center_y - im0.shape[0] / 2  # y coordinate

                        # Update min_distance and min_box_center if current box has shorter distance
                        distance = dx**2 + dy**2  # Euclidean distance squared
                        if distance < min_distance:
                            min_distance = distance
                            dxx = int(dx)
                            dyy = int(dy)
                
                print(dxx,dyy)
                SendInput(mouse_input(1, dxx, dyy))


        


            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

            




        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        keyboard.add_hotkey('q', dataset.stop)


    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    
    




def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "models/cs088.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="screen 0 640 360 640 360", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="mss2test", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


# 设置检查目标
def set_variable(v1,v2):
    global headorbody
    global TorCT
    if v1==1:
        if headorbody == 'head':
            headorbody = 'body'
    if v2==1:
        if TorCT == 'T':
            TorCT = 'CT'
    print(f"flags set to: {headorbody,TorCT}")


def main(opt):
    global headorbody 
    headorbody ='head'
    global TorCT 
    TorCT ='T'
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # 使用 keyboard.add_hotkey 绑定按键事件到函数
    keyboard.add_hotkey('o', set_variable, args=(1,0,))
    keyboard.add_hotkey('p', set_variable, args=(0,1,))
    run(**vars(opt))
    # print(headorbody)
    # print(TorCT)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


