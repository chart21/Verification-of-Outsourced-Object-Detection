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

### Contractor and Verifier

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
#### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2


#### Edge TPU drivers (If you are planning to use a Coral Edge USB Accelerator)
https://coral.ai/docs/accelerator/get-started/

### Outsourcer (Raspberry Pi)
Install all required python dependecies. Installing open-cv can be done with this guide: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html




### Downloading Official Pre-trained Weights
The Edge TPU model is already contained in this repository because it is only 6MB in size.

YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes we will use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

### Using Custom Trained YOLOv4 Weights

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.

The only change within the code you need to make in order for your custom model to work is on line 14 of 'core/config.py' file.
Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)
<p align="center"><img src="data/helpers/custom_config.png" width="640"\></p>

<strong>Note:</strong> If you are using the pre-trained yolov4 then make sure that line 14 remains <strong>coco.names</strong>.

## YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.

#### Convert darknet weights to tensorflow
##### yolov4

```bash
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 
```

###### yolov4-tiny

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

## Execution
After all machines are setup and atleast one model is saved we can start executing the program. First open **parameters.py** and change IPs according to the local IPs of your machines. You can also make changes to the model used, whether you want to use Merkle Trees, sampling invertvals, maximum allowed loss rates and much more. Note that OutsourceContract and VerifierContract has to be identical on your machine running the outsourcer and your machines running contractor/verifier respectively.

Afterwards you can start **outsourcer.py** on the Raspberry Pi and either **contractor.py**, **contractor_EdgeTpu.py**, **contractor_with_multithreading.py**, or **contractor_EdgeTpu_with_multithreading.py** depending on which version you want to use.

If everything was set up correctly, the outsourcer will start sending live webcam video to contractor and verifier and receive results. You can cancel the contract according to custom if you press **q** in the CV2 output of verifier or contractor.  

## Supported Contract Violations
Contract violations are distinguished bwetween (1) Quality of Experience (QOE) violations due to timeouts, or not receiving/acknowlidging enough outputs, and (2) Malicious behavior. Consequences of QOE violations can be blacklisting, and bad reviews (if Merkle Trees are used also refusing payment of last interval). Consequences of malicious behavior can be fines, and refusal of payment. Every party that is accused of malicious behavior has the right to contest if additional verifiers are available within a deadline.

### QOE Viaolations
#### Otsourcer perspective
1. Contractor did not connect in time
2. Verfier did not connect in time
3. Contractor response is ill formated
4. Verifier response is ill formated
5. Contractor singature does not match response
6. Verifier singature does not match response
7. Contractor response delay rate is too high
8. Verifier has failed to process enough samples in time
9. No root hash received for current interval in time
10. Merkle tree leaf node does not match earlier sent response
11. Contractor signature of challenge response is incorrect
12. Leaf is not contained in Merkle Tree
13. Contractor signature of root hash received at challenge response does not match previous signed root hash
14. Merkle Tree proof of membership challenge response was not received in time

#### Contractor/Verifier perspective
1. Outsourcer signature does not match input
2. Outsourcer did not acknowledge enough ouputs
3. Outsourcer timed out


### Malicious Behaviors
1. Merkle Tree is built on responses unequal to responses of the verifier
2. Contractor output and verifier sample are not equal


## References  

   This repository builds on the following existing repositories:
   
   https://github.com/theAIGuysCode/yolov4-custom-functions - To run Yolov4 with tensorflow
   and get formatted outputs
   
   https://github.com/redlogo/RPi-Stream - To setup a raspberry Pi image stream and use a Coral Edge TPU for inferencing
   
   
