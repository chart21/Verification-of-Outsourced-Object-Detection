# Verification of outsourced Object Detection
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


This project lets you send a digitally signed image stream from an Outsourcer (Raspberry pi) to two machines in the local network. One remote machine acts as a Contractor and one acts as a Verifier. The Contractor receives all images while the Outsourcer only receives random samples. Whenever the Contractor and the Verifier send back a signed object detection result belonging to the same image, the Outsourcer checks if they are equal. At the end of a contract, signatures can be used as a proof to redeem payment or to convict a party of cheating.

 Supported models for object detection on a regular GPU and CPU are Yolov4 and Yolov3 using Tensorflow, TFLite, and TensorRT (only deterministic) as the framework. Tiny weights and custom weights can be used as well.
 
 The supported model for object detection on a Coral USB Accelerator is Mobilenet SSD V2.

 When executing the multithreading version of the scripts, adding signatures to images and responses should not increase the processing time at all as inference is usually the bottleneck of the setup.   





## Demos

### Object Detection using Yolov4 with CPU/GPU
<p align="center"><img src="data/helpers/demo.gif"\></p>

### Whole setup using Yolov4 with CPU/GPU
<p align="center"><img src="data/demo/Yolo-setup.gif"\></p>

### Whole setup using Mobilenet SSD V2 with Coral Edge USB Accelerator
<p align="center"><img src="data/demo/EdgeTpu-Setup.gif"\></p>

## Supported Contract Violations
Contract violations are distinguished between (1) Quality of Experience (QoE) Violations due to timeouts, or not receiving/acknowledging enough outputs, and (2) Malicious Behavior. Consequences of QOE violations can be blacklisting, and bad reviews (if Merkle Trees are used also refusing payment of last interval). Consequences of malicious behavior can be fines, and refusal of payment. Every party that is accused of malicious behavior has the right to contest if additional Verifiers are available within a deadline.


### QoE Violations
#### Outsourcer perspective
1. Contractor did not connect in time
2. Verifier did not connect in time
3. Contractor response is ill formated
4. Verifier response is ill formated
5. Contractor signature does not match response
6. Verifier signature does not match response
7. Contractor response delay rate is too high
8. Verifier has failed to process enough samples in time
9. No root hash received for current interval in time
10. Merkle tree leaf node does not match earlier sent response
11. Contractor signature of challenge response is incorrect
12. Leaf is not contained in Merkle Tree
13. Contractor signature of root hash received at challenge response does not match previous signed root hash
14. Merkle Tree proof of membership challenge-response was not received in time

#### Contractor/Verifier perspective
1. Outsourcer signature does not match input
2. Outsourcer did not acknowledge enough outputs
3. Outsourcer timed out


### Malicious Behaviors
1. Merkle Tree is built on responses unequal to responses of the Verifier
2. Contractor output and Verifier sample are not equal

By changing **parameters.py** you can modify the thresholds of QOE violations. 

## Getting Started

### Contractor and Verifier

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f verified-outsourcing-cpu.yml
conda activate verified-outsourcing-cpu

# Tensorflow GPU
conda env create -f verified-outsourcing-gpu.yml
conda activate verified-outsourcing-gpu
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


#### Edge TPU drivers (If you are planning to use a Coral USB Accelerator)
https://coral.ai/docs/accelerator/get-started/

### Outsourcer (Raspberry Pi)
Install all required python dependencies. Installing open-cv can be done with this guide: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html




### Downloading Official Pre-trained Weights
The Edge TPU model is already contained in this repository because it is only 6MB in size.

YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes, you can use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download the file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

### Using Custom Trained YOLOv4 Weights

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.

The only change within the code you need to make for your custom model to work is on line 14 of 'core/config.py' file.
Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)
<p align="center"><img src="data/helpers/custom_config.png" width="640"\></p>

<strong>Note:</strong> If you are using the pre-trained yolov4 then make sure that line 14 remains <strong>coco.names</strong>.

### YOLOv4 Using Tensorflow (tf, .pb model)
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




### Command Line Args Reference

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
After all machines are set up and at least one model is saved we can start executing the program. First, open **parameters.py** and change IPs according to the local IPs of your machines. You can also make changes to the model used, whether you want to use Merkle Trees, sampling intervals, maximum allowed loss rates, and much more. Note that OutsourceContract and VerifierContract have to be identical on your machine running the Outsourcer and your machines running the Contractor and the Verifier respectively.

Afterward, you can start **Outsourcer.py** on the Raspberry Pi and either **Contractor.py**, **Contractor_EdgeTpu.py**, **Contractor_with_multithreading.py**, or **Contractor_EdgeTpu_with_multithreading.py** depending on which version you want to use.

If everything was set up correctly, the Outsourcer will start sending a live webcam image stream to the Contractor and sample images to the Verifier. Verifier and Contractor will send back object detection results. All messages sent between machines are signed by the sending entity and verified by the receiving entity using ED25519 signatures of message content, SHA3-256 hash of Verifier Contract or Outsource Contract, and additional information depending on the setup. You can cancel the contract according to custom if you press **q** in the CV2 output window of Verifier or Contractor.  



## References  

   This repository re-uses components of the following existing repositories:
   
   https://github.com/theAIGuysCode/yolov4-custom-functions - To run Yolov4 with tensorflow and get formatted outputs
   
   https://github.com/redlogo/RPi-Stream - To setup a Raspberry Pi image stream and use a Coral Edge Accelerator for inferencing
   
   
