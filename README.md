# yolov10 AIMBOT项目
唐硕10225101447
核心代码位于detect.py，对1080p显示器中心区域进行推理，默认不追踪，按下p切换是否追踪，第一次按下追踪T阵营恐怖分子，按下l切换追踪目标即CT和T阵营目标切换，按下q停止项目


mouseinput.py中定义了鼠标移动的方法
加载模型，循环截图，推理图片，遍历boxes，方法查看都依照于yolov10-main\ultralytics\engine\results.py里类的定义，检查是否为想要追踪的类别，检查锚框的长宽比（排除尸体），找到所有符合的类别的目标中距离最近的并对其锁定

在定义距离时对头部距离加上乘子0.6可以实现智能追踪头部，如果头部太小检测不到也会顺势锁定身体，可以解决头部锁定成功率低，也避免了放着近距离但检测不到头的敌人不打扭头锁更遥远目标的头这种异常行为

基于kaggle的特斯拉P100时，基于yolov10n只需要1.8毫秒推理时间，最大的yolov10b也只需要9.6ms，p100的算力相较于4060等新架构卡并不高，理论上有极高的实用可能性，只截取640*640的图像，可以实现上百帧的推理；但是我只有cpu可供推理，基于r5 5600，在单纯推理下耗时约80ms一个循环，开启游戏和OBS后就要280ms一个循环，录制视频效果极差；试图去网吧运行，但是因为网吧电脑重启之后的重置，无法安装cuda，遂失败。

yolov10去除了NMS,使用两个v8的损失函数求和，对比v5的性能提升：
![v10n](v10b.jpg)
![v5s训练结束没有输出推理时间不好横向比较](v5s.jpg)

在注释代码中可以调用实时显示检测图像的部分，已被注释掉；该模型会检测尸体，但是已经在代码中通过检查锚框长宽比来排除，而不必标注负样本，猜测原因可能来自过强的泛化能力或者缺少yolov5自动计算锚框最适合大小的部分（不确定

<!-- # [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)


Official PyTorch implementation of **YOLOv10**.

<p align="center">
  <img src="figures/latency.svg" width=48%>
  <img src="figures/params.svg" width=48%> <br>
  Comparisons with others in terms of latency-accuracy (left) and size-accuracy (right) trade-offs.
</p>

[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458).\
Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding\
[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2405.14458) <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/jameslahm/yolov10-665b0d90b0b5bb85129460c2) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/jameslahm/YOLOv10)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kadirnar/Yolov10)  [![Transformers.js Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers.js-blue)](https://huggingface.co/spaces/Xenova/yolov10-web) [![LearnOpenCV](https://img.shields.io/badge/BlogPost-blue?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAMAAAC67D%2BPAAAALVBMVEX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F6%2Bfn6%2Bvq3y%2BJ8rOFSne9Jm%2FQcOlr5DJ7GAAAAB3RSTlMAB2LM94H1yMxlvwAAADNJREFUCFtjZGAEAob%2FQMDIyAJl%2FmFkYmEGM%2F%2F%2BYWRmYWYCMv8BmSxYmUgKkLQhGYawAgApySgfFDPqowAAAABJRU5ErkJggg%3D%3D&logoColor=black&labelColor=gray)](https://learnopencv.com/yolov10/) [![Openbayes Demo](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/public/tutorials/im29uYrnIoz) 


<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Over the past years, YOLOs have emerged as the predominant paradigm in the field of real-time object detection owing to their effective balance between computational cost and detection performance. Researchers have explored the architectural designs, optimization objectives, data augmentation strategies, and others for YOLOs, achieving notable progress. However, the reliance on the non-maximum suppression (NMS) for post-processing hampers the end-to-end deployment of YOLOs and adversely impacts the inference latency. Besides, the design of various components in YOLOs lacks the comprehensive and thorough inspection, resulting in noticeable computational redundancy and limiting the model's capability. It renders the suboptimal efficiency, along with considerable potential for performance improvements. In this work, we aim to further advance the performance-efficiency boundary of YOLOs from both the post-processing and the model architecture. To this end, we first present the consistent dual assignments for NMS-free training of YOLOs, which brings the competitive performance and low inference latency simultaneously. Moreover, we introduce the holistic efficiency-accuracy driven model design strategy for YOLOs. We comprehensively optimize various components of YOLOs from both the efficiency and accuracy perspectives, which greatly reduces the computational overhead and enhances the capability. The outcome of our effort is a new generation of YOLO series for real-time end-to-end object detection, dubbed YOLOv10. Extensive experiments show that YOLOv10 achieves the state-of-the-art performance and efficiency across various model scales. For example, our YOLOv10-S is 1.8$\times$ faster than RT-DETR-R18 under the similar AP on COCO, meanwhile enjoying 2.8$\times$ smaller number of parameters and FLOPs. Compared with YOLOv9-C, YOLOv10-B has 46\% less latency and 25\% fewer parameters for the same performance.
</details>

## Notes
- 2024/05/31: Please use the [exported format](https://github.com/THU-MIG/yolov10?tab=readme-ov-file#export) for benchmark. In the non-exported format, e.g., pytorch, the speed of YOLOv10 is biased because the unnecessary `cv2` and `cv3` operations in the `v10Detect` are executed during inference.
- 2024/05/30: We provide [some clarifications and suggestions](https://github.com/THU-MIG/yolov10/issues/136) for detecting smaller objects or objects in the distance with YOLOv10. Thanks to [SkalskiP](https://github.com/SkalskiP)!
- 2024/05/27: We have updated the [checkpoints](https://huggingface.co/collections/jameslahm/yolov10-665b0d90b0b5bb85129460c2) with class names, for ease of use.

## UPDATES 🔥
- 2024/06/01: Thanks to [ErlanggaYudiPradana](https://github.com/rlggyp) for the integration with [C++ | OpenVINO | OpenCV](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference)
- 2024/06/01: Thanks to [NielsRogge](https://github.com/NielsRogge) and [AK](https://x.com/_akhaliq) for hosting the models on the HuggingFace Hub!
- 2024/05/31: Build [yolov10-jetson](https://github.com/Seeed-Projects/jetson-examples/blob/main/reComputer/scripts/yolov10/README.md) docker image by [youjiang](https://github.com/yuyoujiang)!
- 2024/05/31: Thanks to [mohamedsamirx](https://github.com/mohamedsamirx) for the integration with [BoTSORT, DeepOCSORT, OCSORT, HybridSORT, ByteTrack, StrongSORT using BoxMOT library](https://colab.research.google.com/drive/1-QV2TNfqaMsh14w5VxieEyanugVBG14V?usp=sharing)!
- 2024/05/31: Thanks to [kaylorchen](https://github.com/kaylorchen) for the integration with [rk3588](https://github.com/kaylorchen/rk3588-yolo-demo)!
- 2024/05/30: Thanks to [eaidova](https://github.com/eaidova) for the integration with [OpenVINO™](https://github.com/openvinotoolkit/openvino_notebooks/blob/0ba3c0211bcd49aa860369feddffdf7273a73c64/notebooks/yolov10-optimization/yolov10-optimization.ipynb)!
- 2024/05/29: Add the gradio demo for running the models locally. Thanks to [AK](https://x.com/_akhaliq)!
- 2024/05/27: Thanks to [sujanshresstha](sujanshresstha) for the integration with [DeepSORT](https://github.com/sujanshresstha/YOLOv10_DeepSORT.git)!
- 2024/05/26: Thanks to [CVHub520](https://github.com/CVHub520) for the integration into [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)!
- 2024/05/26: Thanks to [DanielSarmiento04](https://github.com/DanielSarmiento04) for integrate in [c++ | ONNX | OPENCV](https://github.com/DanielSarmiento04/yolov10cpp)!
- 2024/05/25: Add [Transformers.js demo](https://huggingface.co/spaces/Xenova/yolov10-web) and onnx weights(yolov10[n](https://huggingface.co/onnx-community/yolov10n)/[s](https://huggingface.co/onnx-community/yolov10s)/[m](https://huggingface.co/onnx-community/yolov10m)/[b](https://huggingface.co/onnx-community/yolov10b)/[l](https://huggingface.co/onnx-community/yolov10l)/[x](https://huggingface.co/onnx-community/yolov10x)). Thanks to [xenova](https://github.com/xenova)!
- 2024/05/25: Add [colab demo](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s), [HuggingFace Demo](https://huggingface.co/spaces/kadirnar/Yolov10), and [HuggingFace Model Page](https://huggingface.co/kadirnar/Yolov10). Thanks to [SkalskiP](https://github.com/SkalskiP) and [kadirnar](https://github.com/kadirnar)! 

## Performance
COCO

| Model | Test Size | #Params | FLOPs | AP<sup>val</sup> | Latency |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| [YOLOv10-N](https://huggingface.co/jameslahm/yolov10n) |   640  |     2.3M    |   6.7G   |     38.5%     | 1.84ms |
| [YOLOv10-S](https://huggingface.co/jameslahm/yolov10s) |   640  |     7.2M    |   21.6G  |     46.3%     | 2.49ms |
| [YOLOv10-M](https://huggingface.co/jameslahm/yolov10m) |   640  |     15.4M   |   59.1G  |     51.1%     | 4.74ms |
| [YOLOv10-B](https://huggingface.co/jameslahm/yolov10b) |   640  |     19.1M   |  92.0G |     52.5%     | 5.74ms |
| [YOLOv10-L](https://huggingface.co/jameslahm/yolov10l) |   640  |     24.4M   |  120.3G   |     53.2%     | 7.28ms |
| [YOLOv10-X](https://huggingface.co/jameslahm/yolov10x) |   640  |     29.5M    |   160.4G   |     54.4%     | 10.70ms |

## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```
## Demo
```
python app.py
# Please visit http://127.0.0.1:7860
```

## Validation
[`yolov10n`](https://huggingface.co/jameslahm/yolov10n)  [`yolov10s`](https://huggingface.co/jameslahm/yolov10s)  [`yolov10m`](https://huggingface.co/jameslahm/yolov10m)  [`yolov10b`](https://huggingface.co/jameslahm/yolov10b)  [`yolov10l`](https://huggingface.co/jameslahm/yolov10l)  [`yolov10x`](https://huggingface.co/jameslahm/yolov10x)  
```
yolo val model=jameslahm/yolov10{n/s/m/b/l/x} data=coco.yaml batch=256
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.val(data='coco.yaml', batch=256)
```


## Training 
```
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='coco.yaml', epochs=500, batch=256, imgsz=640)
```

## Push to hub to 🤗

Optionally, you can push your fine-tuned model to the [Hugging Face hub](https://huggingface.co/) as a public or private model:

```python
# let's say you have fine-tuned a model for crop detection
model.push_to_hub("<your-hf-username-or-organization/yolov10-finetuned-crop-detection")

# you can also pass `private=True` if you don't want everyone to see your model
model.push_to_hub("<your-hf-username-or-organization/yolov10-finetuned-crop-detection", private=True)
```

## Prediction
Note that a smaller confidence threshold can be set to detect smaller objects or objects in the distance. Please refer to [here](https://github.com/THU-MIG/yolov10/issues/136) for details.
```
yolo predict model=jameslahm/yolov10{n/s/m/b/l/x}
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.predict()
```

## Export
```
# End-to-End ONNX
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=onnx opset=13 simplify
# Predict with ONNX
yolo predict model=yolov10n/s/m/b/l/x.onnx

# End-to-End TensorRT
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=engine half=True simplify opset=13 workspace=16
# or
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# Predict with TensorRT
yolo predict model=yolov10n/s/m/b/l/x.engine
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.export(...)
```

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
``` -->
