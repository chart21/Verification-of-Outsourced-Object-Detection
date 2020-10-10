"""pub_sub_receive.py -- receive OpenCV stream using PUB SUB."""


from parameters import ParticipantData
from parameters import Parameters
from parameters import OutsourceContract
from parameters import VerifierContract
from parameters import Helperfunctions
#import json
#from merkletools import MerkleTools
import sys
import threadHandler as vss3
#from nacl.signing import SigningKey
#from nacl.signing import VerifyKey
import time
import imagezmq
#import Responder as re
from utilities.stats import MovingAverage
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from core.functions import *
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import os
#from Framsender import FrameSender
# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')

try:
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


#from object_detection.object_detection import Model
#from utilities.render import Render


#from ecdsa import VerifyingKey
#from ecdsa import SigningKey


# Helper class implementing an IO deamon thread


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


    
    #sk = SigningKey(Parameters.private_key_contractor)

    
    framework = Parameters.framework
    
    weights = Parameters.weights
    count = Parameters.count
    info = Parameters.info
    crop = Parameters.crop
    iou = Parameters.iou
    score = Parameters.score
    input_size = Parameters.input_size
    
    
    








    # print(contractHash)

     # configure video stream receiver
    receiver = vss3.ThreadHandler(hostname, port, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk_Bytes, input_size, sendingPort)

    
    
    print('Receiver Initialized')
    #time.sleep(4.0)
    
    #frameSender = FrameSender(hostname, sendingPort, merkle_tree_interval, contractHash)
   
    # configure gpu usage
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # load model
    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
    else:
        saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING])

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

  

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

    moving_average_img_postprocessing_time = MovingAverage(
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
        
        images_data = preprocessOutput[0]
        name = preprocessOutput[1]
        original_image = preprocessOutput[2]
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

        

        # endregion

       # image_preprocessing_time = time.perf_counter()

        # inference

        # region
        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            if model_to_use == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(
                    pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(
                    pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # endregion

        model_inferenced_time = time.perf_counter()

        # image postprocessing

        # region

        #h = time.perf_counter()

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )  # 1.2ms

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape

        bboxes = utils.format_boxes(
            boxes.numpy()[0], original_h, original_w)  # 1ms

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0],
                     valid_detections.numpy()[0]]

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        if crop:
            crop_path = os.path.join(
                os.getcwd(), 'detections', 'crop', image_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                         pred_bbox, crop_path, allowed_classes)

        if count:
            # count objects found
            counted_classes = count_objects(
                pred_bbox, by_class=False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            boxtext, image = utils.draw_bbox(
                original_image, pred_bbox, info, counted_classes, allowed_classes=allowed_classes)
        else:
            boxtext, image = utils.draw_bbox(
                original_image, pred_bbox, info, allowed_classes=allowed_classes)  # 0.5ms

        image = Image.fromarray(image.astype(np.uint8))  # 0.3ms

        # endregion

        if merkle_tree_interval == 0:
            boxtext = 'Image' + str(name[-2]) + ':;' + boxtext
        else:
            boxtext = 'Image' + str(name[-5]) + ':;' + boxtext


        
        

        image_postprocessing_time = time.perf_counter()

        # sign message ->need to add image_count/interval_count (for merkle tree sig), contract hash to output and verificaton
        #frameSender.putData((boxtext, image))
        receiver.putData((boxtext, image, name, image_count))

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

        moving_average_model_inference_time.add(
            model_inferenced_time - thread3_waiting_time)

        moving_average_img_postprocessing_time.add(
            image_postprocessing_time - model_inferenced_time)

        #moving_average_response_signing_time.add(
        #    response_signing_time - image_postprocessing_time)  # adjust for merkle root

        #moving_average_reply_time.add(replied_time - response_signing_time)

        #moving_average_image_show_time.add(image_showed_time - replied_time)

        moving_average_thread4_waiting_time.add(thread4_waiting_time - image_postprocessing_time)

        total_time = moving_average_thread3_waiting_time.get_moving_average() \
            + moving_average_model_inference_time.get_moving_average() \
            + moving_average_img_postprocessing_time.get_moving_average() \
            + moving_average_thread4_waiting_time.get_moving_average() 

        if(image_count == 800):
            a = time.perf_counter()
        if(image_count == 1200):
            a = time.perf_counter() - a
            print(a)

        # terminal prints
        if image_count % 20 == 0:

            print(" total: %4.1fms (%4.1ffps) "
                  " Waiting for Thread 3 (receiving, decoding, verifying, preprocessing) %4.1f (%4.1f%%) "                  
                  " model inference %4.1f (%4.1f%%) "
                  " postprocessing %4.1f (%4.1f%%) "
                  " Waiting for Thread 4 (signing, replying, displaying) %4.1f (%4.1f%%) "

                  % (
                      1000/moving_average_fps.get_moving_average(),
                      moving_average_fps.get_moving_average(),

                      moving_average_thread3_waiting_time.get_moving_average()*1000,
                      moving_average_thread3_waiting_time.get_moving_average() / total_time * 100,

                      moving_average_model_inference_time.get_moving_average()*1000,
                      moving_average_model_inference_time.get_moving_average() / total_time * 100,




                      moving_average_img_postprocessing_time.get_moving_average()*1000,
                      moving_average_img_postprocessing_time.get_moving_average() / total_time * 100,

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
