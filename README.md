# Verification of outsourced Object Detection
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


This project lets you send a digitally signed image stream from an Outsourcer (Raspberry pi) to two machines in the local network. One remote machine acts as a Contractor and the other one acts as a Verifier. The Contractor receives all images while the Outsourcer only receives random samples. Whenever the Contractor and the Verifier send back a signed object detection result belonging to the same image, the Outsourcer checks if both results are equal. At the end of a contract, signatures can be used as a proof to redeem payment or to convict a party of cheating.

 Supported models for object detection on a regular GPU and CPU are Yolov4 and Yolov3 using Tensorflow, TFLite, and TensorRT (only deterministic) as the framework. Tiny weights and custom weights can be used as well.
 
 The supported model for object detection on a Coral USB Accelerator is Mobilenet SSD V2.

 When executing the multithreading version of the scripts, adding signatures to images and responses should not increase the processing time at all as inference is usually the bottleneck of the setup.   




## Demos

### Object Detection using Yolov4 with CPU/GPU
<p align="center"><img src="data/helpers/demo.gif"\></p>

### Whole setup using Yolov4 with CPU/GPU

<p align="center"><img src="data/demo/Yolo-setup.gif"\></p>
Donwload this GIF if you want to see the statistics printed in the consoles.

### Whole setup using Mobilenet SSD V2 with Coral Edge USB Accelerator

<p align="center"><img src="data/demo/EdgeTpu-Setup.gif"\></p>
Donwload this GIF if you want to see the statistics printed in the consoles.





## Supported Contract Violations
Contract violations are distinguished between (1) Quality of Service (QoS) Violations due to timeouts, or not receiving/acknowledging enough outputs, and (2) Malicious Behavior. Consequences of QOS violations can be blacklisting, and bad reviews (if Merkle Trees are used also refusing payment of last interval). Consequences of malicious behavior can be fines, and refusal of payment. Every party that is accused of malicious behavior has the right to contest if additional Verifiers are available within a deadline.


### Qos Violations
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
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2


#### Edge TPU drivers (If you are planning to use a Coral USB Accelerator)
https://coral.ai/docs/accelerator/get-started/

### Outsourcer (Raspberry Pi)
The Outsourcer only relies on the outsourcer folder. Copy it to your Raspberry Pi and install all required python dependencies. Installing open-cv can be done with this guide: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html




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

Afterward, you can start **Outsourcer.py** on the Raspberry Pi and either **Contractor.py**, **Contractor_EdgeTpu.py**, **Contractor_with_multithreading.py**, or **Contractor_EdgeTpu_with_multithreading.py** on the other two machines, depending on which version you want to use. Note that the machine running the Verifier also uses one of the above-listed contractor scripts, but you have to specify in **parameters.py** that it should behave as a Verifier.

If everything was set up correctly, the Outsourcer will start sending a live webcam image stream to the Contractor and sample images to the Verifier. Verifier and Contractor will send back object detection results. All messages sent between machines are signed by the sending entity and verified by the receiving entity using ED25519 signatures of message content, SHA3-256 hash of Verifier Contract or Outsource Contract, and additional information depending on the setup. You can cancel the contract according to custom if you press **q** in the CV2 output window of Verifier or Contractor.  

## Software Architecture
All code is written 100% in Python. Download images for higher quality: https://github.com/chart21/Verification-of-Outsourced-Object-Detection/tree/master/diagrams

### Without multithreading of key tasks
<p align="center"><img src="diagrams/Software Architecture.jpg"\></p>



### With multithreading of key tasks
<p align="center"><img src="diagrams/Software Architecture Multi-threaded.jpg"\></p>



### Without multithreading of key tasks using a Coral USB Accelerator
<p align="center"><img src="diagrams/Software Architecture Multi-threaded EdgeTpu.jpg"\></p>

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;app.diagrams.net\&quot; modified=\&quot;2020-10-21T20:54:43.998Z\&quot; agent=\&quot;5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36\&quot; etag=\&quot;cqqGyG1EzK9Ftf7i4pBV\&quot; version=\&quot;13.8.1\&quot; type=\&quot;device\&quot;&gt;&lt;diagram name=\&quot;Page-1\&quot; id=\&quot;5f0bae14-7c28-e335-631c-24af17079c00\&quot;&gt;7V1bb+O2Ev41fowhkro+Jk7bU6BFg2bRbh9li4l1VrZ8ZDmX8+tL2qIuJBXJulB04mCxsChZtr75OJwZzoxnaLF5+yXxd+vf4wBHM2gEbzN0P4MQWhDO6D8jeD+NANs0TyPPSRhkY8XAY/h/nA0a2eghDPC+cmEax1Ea7qqDq3i7xau0MuYnSfxavewpjqqfuvOfsTDwuPIjcfTvMEjX2SgwjOLEf3D4vM4+2rWyE0t/9eM5iQ/b7PNmED0d/06nNz67V3b9fu0H8WtpCP00Q4skjtPTq83bAkcUXAbb6X0/15zNv3eCt2mbN9zdpvZvP93+8ed+++v7X3D5gp7gTXaXFz86YPYYxy+bvjOAjo+I6U2MGbp7XYcpftz5K3r2lXCCjK3TTUSOAHkpfin2CThJ8VtpKPuSv+B4g9PknVySnb0BDLF3NsC48loIyETZ2LokG8vJBv2MFM/53QtgyIsMmzNwsgWcvi0eZtD2NxSA7XK/Oz68HZFvcrdMyKtn+uoRb4Nw+yw5Ea9+4FTAmnBkR1+u3qOQgJ6gZsSXJ/H8tswHcl7+cUjJbXA2vj9NPWANIyaEqlIi4M8tQUy2REruWEJyugnpT7zC4csXERNi6nkqIQG3WeWQWXNLVTuFOPL3+3BVBRO/hen30ut/qHKaW9nR/Vumq44H76WDB5yE5Clwwsa25Im+lw9Kd6KHxa2OR+/lI/5mtcLax4dkhZuVS+onzzj9CDomKBxU1jRR+CXhWjI9mY0lOPLT8KW6Esoknn3CQxySZ8u5ZXGK2uTV7+nJs3eVlybuRg53I2Q7c9Mr/tzqbU9ACbc90jEHoTtDIZIw9KQZ1oBpBjJpT0+XsHPks4rTAqXJdE2rJN6nSfwDL+IopgzaxlQD3D2FUcQN7YlOocoJ3VvF0beY6Jf7G1i3JMdEQzxFxxm0DoMAb8lYdSH3o/B5SycYoSvlcC9l89FKDbwPGDi8egHiKpDrdSadb+sE+wG5CIgq/zpyHRlnhPB5WYzVmBQnNRHEh2XELIJi3oJmO2PQaf3GT2s2z23JNJdZERYYb56LdsTC36WHBJPBLX6lvmDib7CA9FlwDoKewfszaO6IpjIwJQACazT8PBG/eLNL8H6vL3QAzOHkyDFjpYTcI5102qKGbB1QE0MN1DvWFzVbC65BAbVfn+i7CDZvM2r+3VPYiIcb4WJYNyihqQWWol2v17x1OdRs+byFalEzNZ+3PGquK11dFaM2RJRjyOhEl5BJrXQaIxq5tm+MaUCU8UuTmIbJsaljREOIjhYmn6IoBkAtYvujBNr0CKrliquRgrZWBEQOZ4Z0ZKDJxdR4T2x0+onrRn0cBn5Rz5p3rKXxM5ljPdpGF0CWILfTBknhVxMfcRdv99Ov/fxOITB18HTMc5iPrszPFky52aaW/OYFkx85OpDfGtLwzQ9aWhCZtWLMDQNVLBbTZcftNwe7mx6Q6dDmHT12pS7GB/SqCtUyO1ofHqeZeTNmbPPDEsObXz1jw+bDgqYlUxhqswHYp12TNj6QFLSki6NiSbXwJxnGQbw6bI7PfzbEA8Dn8k58zcooCwmNiB+U4HfibhC+VHC0/3egWYF3NGngJrO+bskV/z3s0/DpvbiglYF4vHuNiXhWct8YdqDNh+8MqawcWbofBGJUZUB5tUkD+YsYDU/hNQmkJFHPaiVRaVoI4ONtA4pTlhfCyYeq9l17CPKkZH/J7mB8CI3bMsIvU+xgPK/HltnsIzmsl+TUNhCpdVoUlCTGSmXsjSdi0Rw+qi76LfcECP+UODGsS9sbP3aWSzOUxMjUOrmOGCZQrkscwUZsGTp0x1w0HTFR/tNHfRs40kzzaSUmrot8yCvc7g6iPzWKcjgDtRrlAOZu6+QqZ+6ZtgldywHH/8fC2G2hL9puxbFwVe+c93ODawNvz7EATWOMLJOKJhEyi49bu2DOzc22MTLL5m6FhFuNHSVzZer6wojZnYNOSwbm2wKacNAU4htd94i5KK2Q7jA6/8TF55jeptlWi5iHimTFboqtUFd028a36XvnZwFbnv2sNj/LE4M7F0E8VBMxUMs8T9zhvgDmIVeHfEpPtAVp3r1PAcuJt9cOPNPQIa3Sm8peUW6b5FkkzdYJ02WaWCcer7E6GieAs3KQ4gy2Y3xC8IzZwrCY5dn2C363sqQC+W1MjqvdNwSEuASd2uHKj26zE5swCOjHSHVFdRdgAHWBBKkDiYGUO+uVXZyxgp7QEJepn08561XpGWKMaXvYLOmujhFTq8Bf/djGrxGdYAEdO6S7Q7o/vfcqbAhtWYRbsbDB11kaUOulwdUseCKWBXWMnQgOmac6dgKlheYc4cbeBAAGF0PKQwHTbSdCaYePCSvwtd4nKGjUf59gtO1DKCkWfkjwLolXmGhRmoQ1rLvSGypNdw7z9a8cd9g+YfLkKzXbr/1BlNX+KwaxRSba6JqXrxRDQBoSVKx7GRKtdK/5lbtgNFCrfc8b2HbFHa0XBqyvbB4rktkbPW0VdG2584VBOb2atsVI50O8T1VbDb2RlE1wxaQc0LetlvK6SF3XvNrVuLmghrkUzeXktl5btXwybE6cDg5v9U68sT96UzzRcu3MQRXxlYHzVfKFoTnowjJbNKGgwxmqFupIQYcL+5l8rcToFByio0HX7gRMfc4No6pCbQuerUJredis4IBe7AKGacwJ5Yu/CkeoV2TCUifPjvWEgA/4mYZq8sk8q676b05xKpPIsRtI1IO2AybtOe1b1epGU+tDmpr23EbC2fMXaT7loNSQQBlRZXVLPYjK9X4htqj+BmPugDcT1dFri4RfrU2+U27n1Vq5whQ96a9egS10hZq+rBdK+r1cC7B5QQEwfak8RKNFAS62cz5svRUOkauVnud755dbZPZtnw9oL51S/3yHK2wZXfN3/CGOT6z5+Y4OANhT/1pKPiOumv+jJili8FetlMwBfy/lk9SO5QqmhdbXKxzMV48BCwqlnV3Lx4Ah3mtsTW+KeSHXFNmWWZMAQZn1rzhtkhH7miM7ckK0tOpJsbClLYh6hIa4GKblTB/D1ETRu5Y3d00j/wMVMrjm3DZdx2Jn7W4rgBBuBEUOkqIVAEly7K9drYSuVjfSHjDqm1ohQ9aJgG9qtYjJ/PNXaZxc21rV7uTLRaq4qxUyWsSEVHe1ktdGq02ERMY5PwP31XtaFTS6oJ5WyJii/r03fpomPiJDdOWUaxKhp5X8Z4NU90jKK0+/fFergiV6d7VCQEwVmq6r1TmoXU5XKwRaeJPKC+BqfmlMsfUB6nvHXkvg6omkcQkcAqKjq74E7hyodDU0gOiCqi2B6w/i5LUVqE21u/ISuJpf3lOte8/x/K4lcPXUuqQSOCQpOp+iBO4c9LRV0KInOEUJXH8op1fTknLvCUrg+iM5eQkcalPz/YnLj5BmZR98mjIyuqev2Xwig2Wr3sViZPrKlUW5h6MJxRoqi4ixaTsjFBYBR2Ty6PT7/CUbLein1w6+bXlz065Qw4Fzo0Q51nexb2XGJIz73BXlzWxjFo0mbAMQ0ISRKi9ojjAUckjO1nAQefMSaz1X/BSj9ClAMRNbtFEba+ntwR9Dr3ICl7Oh3I62mMenP9MfrjcF7vQmBzlMYprsU1xOvJv173GA6RX/Ag==&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>


## Benchmarks

### Key Results

| Participant | Device                   | CPU           | GPU                   | Model                        | Frames per second | Milliseconds per frame | % spent on Network wait | % spent on application processing | % spent on verification scheme | ms spent on verification scheme |
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
| Verifier    | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | 28.75             | 34.78                  | ?                       | 0.64                              | ?                              | ?                               |


### Additional Benchmarks

| Participant | Device                   | CPU           | GPU                   | Model                        | Non-blocking<br>message pattern | Merkle Trees<br>used | Multithreading | Frames per second | Miliseconds per frame | % spent on Network wait | % spent on application processing | % spent on verification scheme | ms spent on verification scheme |
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
| Verifier    | Notebook                 | Core i5 4300U | \-                    | Yolov4 tiny<br>416\*416      | X                               | X                    | ✓              | 6.74              | 148.35                | ?                       | 81.20                             | ?                              | ?                               |
|             |                          |               |                       |                              |                                 |                      |                |                   |                       |                         |                                   |                                |                                 |
| Verifier    | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | X                               | X                    | X              | 28.48             | 35.11                 | 31.20                   | 67.50                             | 1.30                           | 0.46                            |
| Verifier    | Notebook                 | Core i5 4300U | Coral USB Accelerator | Mobilenet SSD V2<br>300\*300 | X                               | X                    | ✓              | 28.75             | 34.78                 | ?                       | 0.64                              | ?                              | ?    


## References  

   This repository re-uses components of the following existing repositories:
   
   https://github.com/theAIGuysCode/yolov4-custom-functions - To run Yolov4 with tensorflow and get formatted outputs
   
   https://github.com/redlogo/RPi-Stream - To setup a Raspberry Pi image stream and use a Coral Edge Accelerator for inferencing
   
   
