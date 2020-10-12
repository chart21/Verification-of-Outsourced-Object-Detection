# Main class of a contractor or verifier using an Edge TPU with the use of threading
# Paramters associated with this class including if this device should act as a contractor or verifier can be set in parameters.py
from parameters import ParticipantData
from parameters import Parameters
from parameters import OutsourceContract
from parameters import VerifierContract
from parameters import Helperfunctions

import sys
import threadHandler_Edgetpu as vss3

import time
import imagezmq

from utilities.stats import MovingAverage

import numpy as np
import cv2
from PIL import Image

from absl import app, flags, logging

import os


from object_detection.object_detection import Model
from utilities.render import Render




def main(_argv):

    # get paramters and contract details

    if Parameters.is_contractor == True:
        vk_Bytes = OutsourceContract.public_key_outsourcer
        merkle_tree_interval = OutsourceContract.merkle_tree_interval
        contractHash = Helperfunctions.hashContract().encode('latin1')
        model_to_use = OutsourceContract.model
        tiny = OutsourceContract.tiny
    else:
        vk_Bytes = VerifierContract.public_key_outsourcer
        contractHash = Helperfunctions.hashVerifierContract().encode('latin1')
        model_to_use = VerifierContract.model
        tiny = VerifierContract.tiny       
        merkle_tree_interval = 0


    port = Parameters.port_outsourcer
    sendingPort = Parameters.sendingPort
    hostname = Parameters.ip_outsourcer  # Use to receive from other computer
    minimum_receive_rate_from_contractor = Parameters.minimum_receive_rate_from_contractor

    dont_show = Parameters.dont_show

    framework = Parameters.framework

    weights = Parameters.weights
    count = Parameters.count
    info = Parameters.info
    crop = Parameters.crop
    iou = Parameters.iou
    score = Parameters.score
    input_size = Parameters.input_size    

    edgeTPU_model_path = Parameters.edgeTPU_model_path
    edgeTPU_label_path = Parameters.edgeTPU_label_Path
    edgeTPU_confidence_level = Parameters.EdgeTPU_confidence_level 
    


     # configure thread handler to handle T2 (receiving), T3 (decompressing, verifying), and T4 (postprocssing, signing, sending, displaying) 

    receiver = vss3.ThreadHandler(hostname, port, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk_Bytes, input_size, sendingPort)
    

    # configure model   
    
    model = Model()
    model.load_model(edgeTPU_model_path)
    model.load_labels(edgeTPU_label_path)
    model.set_confidence_level(edgeTPU_confidence_level)
    
    
    print('Receiver Initialized')
   
    
    # configure and iniitialize statistic variables
   
    moving_average_points = 50

    
    moving_average_fps = MovingAverage(moving_average_points)
    
    moving_average_thread3_waiting_time = MovingAverage(moving_average_points)
    moving_average_thread4_waiting_time = MovingAverage(moving_average_points)

    
    moving_average_model_inference_time = MovingAverage(moving_average_points)

    moving_average_img_preprocessing_time = MovingAverage(
        moving_average_points)


    image_count = 0

    a = 0
    b = 0

    while True:

        start_time = time.perf_counter()

        # receive decompressed image from Thread 3

        preprocessOutput = receiver.receive2()

        thread3_waiting_time = time.perf_counter()        
   
        decompressedImage = preprocessOutput[0]
        name = preprocessOutput[1]

        # image preprocessing

        model.load_image_cv2_backend(decompressedImage)  

        image_preprocessing_time = time.perf_counter()

        # inference
    
        class_ids, scores, boxes = model.inference()  

        model_inferenced_time = time.perf_counter()

        # Transfer inference results to thread 4, wait if it is not finished with last image yet

        receiver.putData((decompressedImage, name, image_count, model.labels, class_ids, boxes))


        thread4_waiting_time = time.perf_counter()



        

        # statistics

        moving_average_fps.add(1 / (thread4_waiting_time - start_time))
        
        moving_average_thread3_waiting_time.add(thread3_waiting_time - start_time)

      

        moving_average_img_preprocessing_time.add(
            image_preprocessing_time - thread3_waiting_time)

        moving_average_model_inference_time.add(
            model_inferenced_time - image_preprocessing_time)

        moving_average_thread4_waiting_time.add(thread4_waiting_time - model_inferenced_time)

        total_time = moving_average_thread3_waiting_time.get_moving_average() \
            + moving_average_model_inference_time.get_moving_average() \
            + moving_average_img_preprocessing_time.get_moving_average() \
            + moving_average_thread4_waiting_time.get_moving_average() 

        # count seconds it takes to process 400 images after a 800 frames warm-up time
        if(image_count == 800):
            a = time.perf_counter()
        if(image_count == 1200):
            a = time.perf_counter() - a
            print(a)

        # terminal prints
        if image_count % 20 == 0:

            print(" total: %4.1fms (%4.1ffps) "
                  " Waiting for Thread 3 (receiving, decoding, verifying) %4.1f (%4.1f%%) " 
                  " preprocessing %4.1f (%4.1f%%) "                 
                  " model inference %4.1f (%4.1f%%) "                  
                  " Waiting for Thread 4 (postprocessing, signing, replying, displaying) %4.1f (%4.1f%%) "

                  % (
                      1000/moving_average_fps.get_moving_average(),
                      moving_average_fps.get_moving_average(),

                      moving_average_thread3_waiting_time.get_moving_average()*1000,
                      moving_average_thread3_waiting_time.get_moving_average() / total_time * 100,

                      moving_average_img_preprocessing_time.get_moving_average()*1000,
                      moving_average_img_preprocessing_time.get_moving_average() / total_time * 100,


                      moving_average_model_inference_time.get_moving_average()*1000,
                      moving_average_model_inference_time.get_moving_average() / total_time * 100,




              
                      moving_average_thread4_waiting_time.get_moving_average()*1000,
                      moving_average_thread4_waiting_time.get_moving_average() / total_time * 100), end='\r')

        # counter
        image_count += 1



if __name__ == '__main__':
    app.run(main)
