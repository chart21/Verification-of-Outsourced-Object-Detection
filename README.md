# Verification of outsourced Object Detection
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


This project lets you send a digitally signed image stream from an Outsourcer (Raspberry pi) to two machines in the local network. One remote machine acts as a Contractor, and the other one acts as a Verifier. The Contractor receives all images, while the Outsourcer only receives random samples. Whenever the Contractor and the Verifier send back a signed object detection result belonging to the same image, the Outsourcer checks if both results are equal. At the end of a contract, signed messages can be used as proof to redeem payment or to convict a party of cheating.

 Supported models for object detection on a regular GPU and CPU are Yolov4 and Yolov3 using Tensorflow, TFLite, and TensorRT (only deterministic) as the framework. Tiny weights and custom weights can be used as well.
 
 The supported model for object detection on a Coral USB Accelerator is Mobilenet SSD V2.

The verification scheme takes less than one millisecond per frame, even on low-end hardware, mainly caused by signing all messages sent and verifying all messages received. When executing the multithreading version of the scripts, adding signatures to images and responses should not increase the overall processing time at all as inference is usually the bottleneck of the setup. 




## Demos

### Object Detection using Yolov4 with CPU/GPU
<p align="center"><img src="data/helpers/demo.gif"\></p>

### Whole setup using Yolov4 with CPU/GPU

<p align="center"><img src="data/demo/Yolo-setup.gif"\></p>
Download this GIF if you want to see the statistics printed in the consoles.

### Whole setup using Mobilenet SSD V2 with Coral Edge USB Accelerator

<p align="center"><img src="data/demo/EdgeTpu-Setup.gif"\></p>
Download this GIF if you want to see the statistics printed in the consoles.





## Supported Contract Violations
Contract violations are distinguished between (1) Quality of Service (QoS) Violations due to timeouts, or not receiving/acknowledging enough outputs, and (2) Dishonest Behavior. Consequences of QoS violations can be blacklisting and bad reviews. Consequences of dishonest behavior can be fines and refusal of payment. Every party accused of dishonest behavior has the right to contest if additional Verifiers are available within a deadline. Our verification scheme predicts or detects 11 different types of protocol violations with high confidence.


|Type of Violation               |Referred Number|Description                                                                  |Techniques                                                                         |Confidence                                           |
|--------------------------------|--------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------|
|Dishonest Behavior by Individual|1             |Contractor sends back false response to save resources                      |Sampling-based re-execution, utilization of a third-party Verifier if required     |>99% (small sampling size)/ <br /> 100% (sampling rate of 1)|
|                                |2             |Verifier sends back false response to save resources                         |Contestation                                                                       |100%                                                 |
|                                |3             |Outsourcer sends different input to Contractor and Verifier to refuse payment|Digital Signatures (signature chain), Contestation                                 |100%                                                 |
|                                |4             |Contractor or Verifier tries to avoid global penalties                       |Digital Signatures                                                                 |100%                                                 |
|                                |5             |Participant refuses to pay even if obliged to by the protocol                |TTP or Blockchain that is authorized to conduct payment on behalf of another entity|100%                                                 |
|Dishonest Behavior via Collusion|6             |Outsourcer and Verifier collude to refuse payment and save resources         |Randomization, Game-theoretic incentives, Contestation                             |100%                                                 |
|                                |7             |Contractor and Verifier collude to save resources                            |Randomization, Game-theoretic incentives                                           |High confidence                                      |
|QoS Violation                   |8             |Timeout                                                                      |Blacklisting, Review system,  Contract abortion                                    |100%                                                 |
|                                |9             |Low Response Rate                                                            |Same as Nr. 8                                                                      |100%                                                 |
|                                |10            |High Response Time                                                           |Same as Nr. 8                                                                      |100%                                                 |
|External Threat                 |11            |Message Tampering                                                            |Digital Signatures                                                                 |100%                                                 |



To detect dishoenst behavior or QoS violations, our scripts check the following exit conditions during execution.

### Qos Violations
#### Outsourcer perspective
1. Contractor did not connect in time
2. Verifier did not connect in time
3. Contractor response is ill formated
4. Verifier response is ill formated
5. Contractor signature does not match its response
6. Verifier signature does not match its response
7. Contractor response delay rate is too high
8. Verifier has failed to process enough samples in time
9. No root hash received for the current interval in time
10. Merkle tree leaf node does not match earlier sent response
11. Contractor signature of challenge-response is incorrect
12. Leaf is not contained in Merkle Tree
13. Contractor signature of root hash received at challenge response does not match previously signed root hash
14. Merkle Tree proof of membership challenge-response was not received in time

#### Contractor/Verifier perspective
1. Outsourcer signature does not match the input
2. Outsourcer did not acknowledge enough outputs
3. Outsourcer timed out


### Dishonest Behaviors
#### Outsourcer perspective
1. Merkle Tree of Contractor is built on responses unequal to responses of the Verifier
2. Contractor response and Verifier sample are not equal

By changing **parameters.py** you can modify the thresholds of QoE violations. 

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
Make sure to use CUDA Toolkit version 10.1 as it is the correct version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2


#### Edge TPU drivers (If you are planning to use a Coral USB Accelerator)
https://coral.ai/docs/accelerator/get-started/

### Outsourcer (Raspberry Pi)
The Outsourcer only relies on the outsourcer folder. Copy it to your Raspberry Pi and install all required python dependencies. Installing open-cv can be done with this guide: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html




### Downloading Official Pre-trained Weights
The Edge TPU model is already contained in this repository because it is only 6MB in size.

YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes, you can use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/openid=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download the file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

### Using Custom Trained YOLOv4 Weights

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.

The only change within the code you need to make for your custom model to work is on line 14 of 'core/config.py' file.
Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)
<p align="center"><img src="data/helpers/custom_config.png" width="640"\></p>

<strong>Note:</strong> If you are using the pre-trained yolov4 then make sure that line 14 remains <strong>coco.names</strong>.

### YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first, we convert the .weights into the corresponding TensorFlow model files and then run the model.

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
After each machine is set up and at least one model is saved, you can start executing the program. First, open **parameters.py** and change IPs according to the local IPs of your machines. You can also make changes to the model used, whether you want to use Merkle Trees, sampling intervals, maximum allowed loss rates, and much more. Note that OutsourceContract and VerifierContract have to be identical on your machine running the Outsourcer and your machines running the Contractor and the Verifier, respectively.

Afterward, you can start **Outsourcer.py** on the Raspberry Pi and either **Contractor.py**, **Contractor_EdgeTpu.py**, **Contractor_with_multithreading.py**, or **Contractor_EdgeTpu_with_multithreading.py** on the other two machines, depending on which version you want to use. Note that the machine running the Verifier also uses one of the above-listed contractor scripts, but you have to specify in **parameters.py** that it should behave as a Verifier.

If everything was set up correctly, the Outsourcer will start sending a live webcam image stream to the Contractor and sample images to the Verifier. Verifier and Contractor will send back object detection results. All messages sent between machines are signed by the sending entity and verified by the receiving entity using ED25519 signatures of message content, SHA3-256 hash of Verifier Contract or Outsource Contract, and additional information depending on the setup. You can cancel the contract according to custom if you press **q** in the CV2 output window of Verifier or Contractor.  

## Software Architecture
All code is written 100% in Python. Download images for higher quality: https://github.com/chart21/Verification-of-Outsourced-Object-Detection/tree/master/diagrams

### Without multithreading of key tasks
<p align="center"><img src="diagrams/Software Architecture.jpg"\></p>



### With multithreading of key tasks
<p align="center"><img src="diagrams/Software Architecture Multi-threaded.jpg"\></p>



### With multithreading of key tasks using a Coral USB Accelerator
<p align="center"><img src="diagrams/Software Architecture Multi-threaded EdgeTpu.jpg"\></p>










## Benchmarks

### Key Results

| Participant | Device                   | CPU           | GPU                   | Model                        | Frames per second | Milliseconds per frame | % spent on network wait | % spent on application processing | % spent on verification scheme | ms spent on verification scheme |
| ----------- | ------------------------ | ------------- | --------------------- | ---------------------------- | ----------------- | ---------------------- | ----------------------- | --------------------------------- | ------------------------------ | ------------------------------- |
| Outsourcer  | Raspberry Pi<br>Model 4B |               |                       | Mobilenet SSD V2<br>300\*300 | 236.00            | 4.24                   | 0.00                    | 78.70                             | 21.30                          | 0.90                            |
| Outsourcer  | Raspberry Pi<br>Model 4B |               |                       | Yolov4 tiny<br>416\*416      | 146.90            | 6.81                   | 0.00                    | 85.10                             | 14.90                          | 1.01                            |
|             |                          |               |                       |                              |                   |                        |                         |                                   |                                |                                 |
| Contractor  | Desktop PC               | Core i7 3770K | GTX 970               | Yolov4 tiny<br>416\*416      | 68.06             | 14.69                  | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
|             |                          |               |                       |                              |                   |                        |                         |                                   |                                |                                 |
| Contractor  | Desktop PC               | Core i7 3770K | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | 63.59             | 15.73                  | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
|             |                          |               |                       |                              |                   |                        |                         |                                   |                                |                                 |
| Contractor  | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | 49.30             | 20.28                  | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
|             |                          |               |                       |                              |                   |                        |                         |                                   |                                |                                 |
| Verifier    | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | 28.75             | 34.78                  | -                       | 0.64                              | -                              | -                               |


### Additional Benchmarks

| Participant | Device                   | CPU           | GPU                   | Model                        | Non-blocking<br>message pattern | Merkle Trees<br>used | Multithreading | Frames per second | Milliseconds per frame | % spent on network wait | % spent on application processing | % spent on verification scheme | ms spent on verification scheme |
| ----------- | ------------------------ | ------------- | --------------------- | ---------------------------- | ------------------------------- | -------------------- | -------------- | ----------------- | --------------------- | ----------------------- | --------------------------------- | ------------------------------ | ------------------------------- |
| Outsourcer  | Raspberry Pi<br>Model 4B |               |                       | Mobilenet SSD V2<br>300\*300 | ✓                               | X                    | X              | 236.00            | 4.24                  | 0.00                    | 78.70                             | 21.30                          | 0.90                            |
| Outsourcer  | Raspberry Pi<br>Model 4B |               |                       | Mobilenet SSD V2<br>300\*300 | ✓                               | ✓                    | X              | 235.10            | 4.25                  | 0.00                    | 78.40                             | 21.60                          | 0.92                            |
| Outsourcer  | Raspberry Pi<br>Model 4B |               |                       | Yolov4 tiny<br>416\*416      | ✓                               | X                    | X              | 135.60            | 7.37                  | 0.00                    | 81.10                             | 18.90                          | 1.39                            |
| Outsourcer  | Raspberry Pi<br>Model 4B |               |                       | Yolov4 tiny<br>416\*416      | ✓                               | ✓                    | X              | 146.90            | 6.81                  | 0.00                    | 85.10                             | 14.90                          | 1.01                            |
|             |                          |               |                       |                              |                                 |                      |                |                   |                       |                         |                                   |                                |                                 |
| Contractor  | Desktop PC               | Core i7 3770K | GTX 970               | Yolov4 tiny<br>416\*416      | ✓                               | X                    | X              | 46.62             | 21.45                 | 0.00                    | 98.30                             | 1.70                           | 0.36                            |
| Contractor  | Desktop PC               | Core i7 3770K | GTX 970               | Yolov4 tiny<br>416\*416      | ✓                               | ✓                    | X              | 46.22             | 21.64                 | 0.00                    | 98.60                             | 1.40                           | 0.30                            |
| Contractor  | Desktop PC               | Core i7 3770K | GTX 970               | Yolov4 tiny<br>416\*416      | ✓                               | X                    | ✓              | 68.03             | 14.70                 | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
| Contractor  | Desktop PC               | Core i7 3770K | GTX 970               | Yolov4 tiny<br>416\*416      | ✓                               | ✓                    | ✓              | 68.06             | 14.69                 | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
|             |                          |               |                       |                              |                                 |                      |                |                   |                       |                         |                                   |                                |                                 |
| Contractor  | Desktop PC               | Core i7 3770K | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | ✓                               | X                    | X              | 57.22             | 17.48                 | 0.00                    | 98.10                             | 1.90                           | 0.33                            |
| Contractor  | Desktop PC               | Core i7 3770K | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | ✓                               | ✓                    | X              | 57.73             | 17.32                 | 0.00                    | 98.50                             | 1.50                           | 0.26                            |
| Contractor  | Desktop PC               | Core i7 3770K | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | ✓                               | X                    | ✓              | 63.19             | 15.83                 | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
| Contractor  | Desktop PC               | Core i7 3770K | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | ✓                               | ✓                    | ✓              | 63.59             | 15.73                 | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
|             |                          |               |                       |                              |                                 |                      |                |                   |                       |                         |                                   |                                |                                 |
| Contractor  | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | ✓                               | X                    | ✓              | 49.30             | 20.28                 | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
| Contractor  | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | ✓                               | ✓                    | ✓              | 49.26             | 20.30                 | 0.00                    | 100.00                            | 0.00                           | 0.00                            |
|             |                          |               |                       |                              |                                 |                      |                |                   |                       |                         |                                   |                                |                                 |
| Verifier    | Notebook                 | Core i5 4300U | \-                    | Yolov4 tiny<br>416\*416      | X                               | X                    | X              | 6.73              | 148.50                | 10.10                   | 89.50                             | 0.40                           | 0.59                            |
| Verifier    | Notebook                 | Core i5 4300U | \-                    | Yolov4 tiny<br>416\*416      | X                               | X                    | ✓              | 6.74              | 148.35                | -                       | 81.20                             | -                              | -                               |
|             |                          |               |                       |                              |                                 |                      |                |                   |                       |                         |                                   |                                |                                 |
| Verifier    | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | X                               | X                    | X              | 28.48             | 35.11                 | 31.20                   | 67.50                             | 1.30                           | 0.46                            |
| Verifier    | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | X                               | X                    | ✓              | 28.75             | 34.78                 | -                       | 0.64                              | -                              | -    


## References  

   This repository re-uses components of the following existing repositories:
   
   https://github.com/theAIGuysCode/yolov4-custom-functions - To run Yolov4 with TensorFlow and get formatted outputs
   
   https://github.com/redlogo/RPi-Stream - To set up a Raspberry Pi image stream and use a Coral USB Accelerator for inference
   
   
