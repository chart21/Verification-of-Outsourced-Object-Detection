# Verification of outsourced Object Detection
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


Sending digitally signed image stream from raspberry pi to a pc in the local network using Pub/Sub. The remote PC performs object detection with Tensorflow using Yolov4 or Yolov3 as model. Detected bounding boxes, classes and confidence get signed by the remote PC before sending them back to the raspberry Pi using sockets



Works with YOLOv4, YOLOv4-tiny, YOLOv3, and YOLOv3-tiny using TensorFlow, TFLite and TensorRT (only determnistic).

## Demos

### Object Detection using Yolov4 with CPU/GPU
<p align="center"><img src="data/helpers/demo.gif"\></p>

### Whole setup using Yolov4 with CPU/GPU
<p align="center"><img src="data/demo/Yolo-setup.gif"\></p>

### Whole setup using Mobilenet SSD V2 with Coral Edge USB Accelerator
<p align="center"><img src="data/demo/EdgeTpu-Setup.gif"\></p>




## Getting Started
### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2


### Edge TPU drivers (If you are planning to use a Coral Edge USB Accelerator)
https://coral.ai/docs/accelerator/get-started/

## Downloading Official Pre-trained Weights
YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes we will use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

## Using Custom Trained YOLOv4 Weights

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.

The only change within the code you need to make in order for your custom model to work is on line 14 of 'core/config.py' file.
Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)
<p align="center"><img src="data/helpers/custom_config.png" width="640"\></p>

<strong>Note:</strong> If you are using the pre-trained yolov4 then make sure that line 14 remains <strong>coco.names</strong>.

## YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.

# Convert darknet weights to tensorflow
## yolov4

```bash
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 
```

## yolov4-tiny

```bash
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny
```




## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)

  ```



### References  

   This repository builds on the following existing repositories:
   
   https://github.com/theAIGuysCode/yolov4-custom-functions - To run Yolov4 with tensorflow
   and get formatted outputs
   
   https://github.com/redlogo/RPi-Stream - To setup a raspberry Pi image stream and use a Coral Edge TPU for inferencing
   
   
