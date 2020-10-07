"""pub_sub_receive.py -- receive OpenCV stream using PUB SUB."""


from parameters import ParticipantData
from parameters import Parameters
from parameters import OutsourceContract
from parameters import Helperfunctions
#import json
#from merkletools import MerkleTools
import sys
import videoStramSubscriber3EdgeTpu as vss3
#from nacl.signing import SigningKey
#from nacl.signing import VerifyKey
import time
import imagezmq
#import Responder as re
from utilities.stats import MovingAverage
#from tensorflow.compat.v1 import InteractiveSession
#from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2
from PIL import Image
#from tensorflow.python.saved_model import tag_constants
#from core.functions import *
#from core.yolov4 import filter_boxes
#import core.utils as utils
#from absl.flags import FLAGS
from absl import app, flags, logging
#import tensorflow as tf
import os
#from Framsender import FrameSender
# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from object_detection.object_detection import Model
from utilities.render import Render





#from object_detection.object_detection import Model
#from utilities.render import Render


#from ecdsa import VerifyingKey
#from ecdsa import SigningKey


# Helper class implementing an IO deamon thread


def main(_argv):

    # get paramters and contract details
    vk_Bytes = OutsourceContract.public_key_outsourcer
    merkle_tree_interval = OutsourceContract.merkle_tree_interval
    port = Parameters.port_outsourcer
    sendingPort = Parameters.sendingPort
    hostname = Parameters.ip_outsourcer  # Use to receive from other computer
    minimum_receive_rate_from_contractor = Parameters.minimum_receive_rate_from_contractor

    dont_show = Parameters.dont_show


    contractHash = Helperfunctions.hashContract().encode('latin1')
    #sk = SigningKey(Parameters.private_key_contractor)

    model_to_use = OutsourceContract.model
    framework = Parameters.framework
    tiny = OutsourceContract.tiny
    weights = Parameters.weights
    count = Parameters.count
    info = Parameters.info
    crop = Parameters.crop
    iou = Parameters.iou
    score = Parameters.score
    input_size = Parameters.input_size
    
    
    








    # print(contractHash)

     # configure video stream receiver
    receiver = vss3.VideoStreamSubscriber(hostname, port, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk_Bytes, input_size, sendingPort)
    
    model = Model()
    model.load_model('models_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    model.load_labels('labels_edgetpu/coco_labels.txt')
    model.set_confidence_level(0.3)
    
    
    print('RPi Stream -> Receiver Initialized')
    #time.sleep(4.0)
    
    #frameSender = FrameSender(hostname, sendingPort, merkle_tree_interval, contractHash)
   

  

    # configure responder
    #responder = re.Responder(hostname, sendingPort)
    #responder = re.Responder(hostname, sendingPort)

    # statistics info
    moving_average_points = 50

    # statistics
    moving_average_fps = MovingAverage(moving_average_points)
    #moving_average_receive_time = MovingAverage(moving_average_points)
    #moving_average_decompress_time = MovingAverage(moving_average_points)

    #moving_average_model_load_image_time = MovingAverage(moving_average_points)
    #moving_average_img_preprocessing_time = MovingAverage(
    #    moving_average_points)

    
    moving_average_thread3_waiting_time = MovingAverage(moving_average_points)
    moving_average_thread4_waiting_time = MovingAverage(moving_average_points)

    
    moving_average_model_inference_time = MovingAverage(moving_average_points)

    moving_average_img_preprocessing_time = MovingAverage(
        moving_average_points)

    #moving_average_reply_time = MovingAverage(moving_average_points)
    #moving_average_image_show_time = MovingAverage(moving_average_points)
    #moving_average_verify_image_sig_time = MovingAverage(moving_average_points)

    #moving_average_response_signing_time = MovingAverage(moving_average_points)

    image_count = 0

    a = 0
    b = 0

    # if merkle_tree_interval > 0:
    #     mt = MerkleTools()
    #     mtOld = MerkleTools()
    #     interval_count = 0
    #     mtOld_leaf_indices = {}
    #     mt_leaf_indices = {}
    #     #rendundancy_counter = 0
    #     #rendundancy_counter2 = 0
    #     current_challenge = 1
    #     merkle_root = ''
    #     #stringsend = ''
    #     last_challenge = 0

    while True:

        start_time = time.perf_counter()

        # receive image

        # region

        # name[:-2] image signature, name
        preprocessOutput = receiver.receive2()

        thread3_waiting_time = time.perf_counter()
        
        #images_data = preprocessOutput[0]
        decompressedImage = preprocessOutput[0]
        name = preprocessOutput[1]
        #original_image = preprocessOutput[2]
        #compressed = preprocessOutput[1]
        #decompressedImage = preprocessOutput[2]


        # if merkle_tree_interval > 0:
        #     outsorucer_signature = name[:-5]
        #     outsourcer_image_count = name[-5]
        #     outsourcer_number_of_outputs_received = name[-4]
        #     outsourcer_random_number = name[-3]
        #     outsourcer_interval_count = name[-2]
        #     outsourcer_time_to_challenge = bool(name[-1])            




        #received_time = time.perf_counter()
        

        # decompress image


        # endregion

        #decompressed_time = time.perf_counter()

        # verify image  (verify if signature matches image, contract hash and image count, and number of outptuts received)


       
       
        #print(name[-2], image_count, name[-3])

        #verify_time = time.perf_counter()

        model.load_image_cv2_backend(decompressedImage)

        # endregion

        image_preprocessing_time = time.perf_counter()

        # inference

        # region
        class_ids, scores, boxes = model.inference()

        # endregion

        model_inferenced_time = time.perf_counter()

        # image postprocessing

        # region

        #h = time.perf_counter()





        # endregion

      


        
        

        #image_postprocessing_time = time.perf_counter()

        # sign message ->need to add image_count/interval_count (for merkle tree sig), contract hash to output and verificaton
        #frameSender.putData((boxtext, image))
        receiver.putData((decompressedImage, name, image_count, model.labels, class_ids, boxes))



        thread4_waiting_time = time.perf_counter()

        #response_signing_time = time.perf_counter()

        # print(response_signing_time- image_postprocessing_time)

        #replied_time = time.perf_counter()

        #image_showed_time = time.perf_counter()

        

        # statistics

        moving_average_fps.add(1 / (thread4_waiting_time - start_time))

        #moving_average_receive_time.add(received_time - start_time)

        #moving_average_decompress_time.add(decompressed_time - received_time)

        #moving_average_verify_image_sig_time.add(
        #   verify_time - decompressed_time)

        #moving_average_img_preprocessing_time.add(
        #    image_preprocessing_time - verify_time)
        
        moving_average_thread3_waiting_time.add(thread3_waiting_time - start_time)

      

        moving_average_img_preprocessing_time.add(
            image_preprocessing_time - thread3_waiting_time)

        moving_average_model_inference_time.add(
            model_inferenced_time - image_preprocessing_time)

        #moving_average_response_signing_time.add(
        #    response_signing_time - image_postprocessing_time)  # adjust for merkle root

        #moving_average_reply_time.add(replied_time - response_signing_time)

        #moving_average_image_show_time.add(image_showed_time - replied_time)

        moving_average_thread4_waiting_time.add(thread4_waiting_time - model_inferenced_time)

        total_time = moving_average_thread3_waiting_time.get_moving_average() \
            + moving_average_model_inference_time.get_moving_average() \
            + moving_average_img_preprocessing_time.get_moving_average() \
            + moving_average_thread4_waiting_time.get_moving_average() 

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

    # except (KeyboardInterrupt, SystemExit):
    #     print('Exit due to keyboard interrupt')
    # except Exception as ex:
    #     print('Python error with no Exception handler:')
    #     print('Traceback error:', ex)
    #     traceback.print_exc()
    # finally:
    #     receiver.close()
    #     sys.exit()


if __name__ == '__main__':
    # try:
    #     app.run(main)
    # except SystemExit:
    #     pass
    app.run(main)
