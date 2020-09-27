"""pub_sub_receive.py -- receive OpenCV stream using PUB SUB."""



import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#from object_detection.object_detection import Model
#from utilities.render import Render
from utilities.stats import MovingAverage

import Responder as re

import imagezmq
import time
from ecdsa import VerifyingKey
from ecdsa import SigningKey


import videoStramSubscriber as vss

import sys

from merkletools import MerkleTools
import json



# Helper class implementing an IO deamon thread


def main(_argv):
    #vk = b'Y\xf8D\xe6o\xf9MZZh\x9e\xcb\xe0b\xb7h\xdb\\\xd7\x80\xd2S\xf5\x81\x92\xe8\x109r*U\xebT\x95\x0c\xf2\xf4(\x13%\x83\xb8\xfa;\xf04\xd3\xfb'
    #vk = b'\x83\xac\xdcq8`\xa35s\n\x82\xf4\x9c\xb0Su\x1e\xe1\xf7M;\x81\x8d\x8aCfIdU\x1e5\xd4\x15W\xef\xb1\xcb\xbd&B\x08\xe5\x86\xac\xc0q\xb2'
    vk = b"'\x83\xac\xdcq8`\xa35s\n\x82\xf4\x9c\xb0Su\x1e\xe1\xf7M;\x81\x8d\x8aCfIdU\x1e5\xd4\x15W\xef\xb1\xcb\xbd&B\x08\xe5\x86\xac\xc0q\xb2"
    vk = VerifyingKey.from_string(vk) #Verifying Key from outsourcer
    vk.precompute()

    sk = b'\x9dy\xd8I\x89\xf3!U\xa8\x19S\xe2\xc3\xd4\x99\x7f\x80\x9a!\x15\x8a\xc6lk'
    sk = SigningKey.from_string(sk) #Singing Key from server

    merkle_tree_interval = 127

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    iou = 0.45
    score = 0.5

    model = 'yolov4'
    framework = ''
    tiny = True
    weights = './checkpoints/yolov4-tiny-416'

    count = False
    dont_show = False
    info = True
    crop = False

   





    
    
    
    #images = FLAGS.images
    #images = []
    #images.append("C:/Users/Kitzbi/Documents/tensorflow yolo/yolov4-custom-functions/data/images/dog.jpg")

   

    # Receive from broadcast
    # There are 2 hostname styles; comment out the one you don't need
    #hostname = "127.0.0.1"  # Use to receive from localhost
    hostname = "192.168.178.34"  # Use to receive from other computer
    port = 5555
    receiver = vss.VideoStreamSubscriber(hostname, port)
    print('RPi Stream -> Receiver Initialized')
    time.sleep(1.0)


    sendingPort = 1234

    responder = re.Responder(hostname, sendingPort)


     # load model
    if framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=weights)
    else:
            saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    
    # statistics info
    moving_average_points = 50

      # statistics
    moving_average_fps = MovingAverage(moving_average_points)
    moving_average_receive_time = MovingAverage(moving_average_points)
    moving_average_decompress_time = MovingAverage(moving_average_points)
    
    
    #moving_average_model_load_image_time = MovingAverage(moving_average_points)
    moving_average_img_preprocessing_time = MovingAverage(moving_average_points)


    moving_average_model_inference_time = MovingAverage(moving_average_points)

    moving_average_img_postprocessing_time = MovingAverage(moving_average_points)

    moving_average_reply_time = MovingAverage(moving_average_points)
    moving_average_image_show_time = MovingAverage(moving_average_points)
    moving_average_verify_image_sig_time = MovingAverage(moving_average_points)

    moving_average_response_signing_time = MovingAverage(moving_average_points)
 
    
    
    
 
    image_count = 0


    a = 0
    b = 0

    if merkle_tree_interval > 0:
        mt = MerkleTools()
        interval_count = 0



    try:
        while True:
            
            start_time = time.perf_counter()
            
            # receive image
            name, compressed = receiver.receive()
            received_time = time.perf_counter()

            # decompress image
            decompressedImage = cv2.imdecode(np.frombuffer(compressed, dtype='uint8'), -1)

            decompressed_time = time.perf_counter()

           # print(name[-1])

            # verify image
            vk.verify( bytes(name[:-1]), compressed)
            verify_time = time.perf_counter()
            
            # image preprocessing  

            
           
            original_image = cv2.cvtColor(decompressedImage, cv2.COLOR_BGR2RGB)
          

            image_data = cv2.resize(original_image, (input_size, input_size)) #0.4ms
           
            image_data = image_data / 255. #2.53ms
           
            # get image name by using split method
            #image_name = image_path.split('/')[-1]
            #image_name = image_name.split('.')[0]

            images_data = []
            
            
            
            for i in range(1):
                images_data.append(image_data)
            
            images_data = np.asarray(images_data).astype(np.float32) #3.15ms
            
            image_preprocessing_time = time.perf_counter()


            # inference 


            if framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if model == 'yolov3' and tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                infer = saved_model_loaded.signatures['serving_default']
                batch_data = tf.constant(images_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            model_inferenced_time = time.perf_counter()


            #image postprocessing

            # run non max suppression on detections


            h = time.perf_counter()

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            ) # 1.2ms
            

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = original_image.shape 

       

            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w) #1ms

            
            
            # hold all detection data in one variable
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]] 

            
            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())


            
            # custom allowed classes (uncomment line below to allow detections for only people)
            #allowed_classes = ['person']

            

            # if crop flag is enabled, crop each detection and save it as new image
            if crop:
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

            if count:
                # count objects found
                counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                boxtext, image = utils.draw_bbox(original_image, pred_bbox, name[-1], info, counted_classes, allowed_classes=allowed_classes)
            else:
                boxtext, image = utils.draw_bbox(original_image, pred_bbox, name[-1], info, allowed_classes=allowed_classes) #0.5ms


           
            image = Image.fromarray(image.astype(np.uint8)) # 0.3ms

           

            image_postprocessing_time = time.perf_counter()
            


            #print(boxtext)

            #sender = imagezmq.ImageSender("tcp://*:{}".format(target_port), REQ_REP=False)
            

            # sign message

            if merkle_tree_interval == 0 :
                sig = sk.sign_deterministic(boxtext.encode('latin1'))
                #sig = list(sig)
                sig = sig.decode('latin1')
                # send reply

                
               
                
            
                responder.respond(boxtext + ';--' + sig)
            else :
                mt.add_leaf(boxtext, True)
                response = boxtext
                if image_count > 0 and image_count % merkle_tree_interval == 0 :
                    mt.make_tree()                    
                    merkle_root = mt.get_merkle_root()
                    sig = sk.sign_deterministic(merkle_root.encode('latin1'))
                    response += ';--' + str(merkle_root) + ';--' + sig.decode('latin1')
                    interval_count += 1
                responder.respond(response)
            #s = socket.socket()
            #s.connect(('192.168.178.34',6001))
            #s
            response_signing_time = time.perf_counter()


            #if(info):
            #    image_hub.send_reply(boxtext)
            #else:
            #    image_hub.send_reply('Ok')
            
            
            #stra = str(pred_bbox).encode()
            #image_hub.send_reply(stra)
            #print(stra)
            #image_hub.send_reply(str(pred_bbox).encode())
            #image_hub.send_reply(bytearray(pred_bbox))

            replied_time = time.perf_counter()


            # display image

            
            if not dont_show:
                #image.show()
                
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                cv2.imshow('raspberrypi', image)
            
             
            
                if cv2.waitKey(1) == ord('q'):
                    break
            
            image_showed_time = time.perf_counter()

            # statistics
            
            moving_average_fps.add(1 / (image_showed_time - start_time))
            
            moving_average_receive_time.add(received_time - start_time)
            
            moving_average_decompress_time.add(decompressed_time - received_time)
            
            moving_average_verify_image_sig_time.add(verify_time - decompressed_time)
            
            moving_average_img_preprocessing_time.add(image_preprocessing_time - verify_time)
            
            moving_average_model_inference_time.add(model_inferenced_time - image_preprocessing_time)

            moving_average_img_postprocessing_time.add(image_postprocessing_time - model_inferenced_time)            

            moving_average_response_signing_time.add(response_signing_time - image_postprocessing_time) #adjust for merkle root

            moving_average_reply_time.add(replied_time - response_signing_time)

            moving_average_image_show_time.add(image_showed_time - replied_time)
            
            total_time = moving_average_receive_time.get_moving_average() \
                        + moving_average_decompress_time.get_moving_average() \
                        + moving_average_verify_image_sig_time.get_moving_average() \
                        + moving_average_img_preprocessing_time.get_moving_average() \
                        + moving_average_model_inference_time.get_moving_average() \
                        + moving_average_img_postprocessing_time.get_moving_average() \
                        + moving_average_response_signing_time.get_moving_average() \
                        + moving_average_reply_time.get_moving_average() \
                        + moving_average_image_show_time.get_moving_average()

            
            if(image_count == 800):
                a = time.perf_counter()
            if(image_count == 1200):
                a = time.perf_counter() -a
                print(a)

            #terminal prints
            if image_count % 20 == 0:
                #print(moving_average_fps)
                #print(decompress_time)
                print(" total: %4.1fms (%4.1ffps) "                  
                    " receiving %4.1f (%4.1f%%) "
                    " decoding %4.1f (%4.1f%%) "
                    " verifying %4.1f (%4.1f%%) "
                    " preprocessing %4.1f (%4.1f%%) "
                    " model inference %4.1f (%4.1f%%) "
                    " postprocessing %4.1f (%4.1f%%) "
                    " signing %4.1f (%4.1f%%) "
                    " replying %4.1f (%4.1f%%) "
                    " display %4.1f (%4.1f%%) "
                    % (
                        1000/moving_average_fps.get_moving_average(),
                        moving_average_fps.get_moving_average(),                   
                        

                        
                        moving_average_receive_time.get_moving_average()*1000,
                        moving_average_receive_time.get_moving_average() / total_time * 100,

                        moving_average_decompress_time.get_moving_average()*1000,
                        moving_average_decompress_time.get_moving_average() / total_time * 100,
                        

                        
                        
                        moving_average_verify_image_sig_time.get_moving_average()*1000,
                        moving_average_verify_image_sig_time.get_moving_average() / total_time * 100,

                        moving_average_img_preprocessing_time.get_moving_average()*1000, 
                        moving_average_img_preprocessing_time.get_moving_average() / total_time * 100,
                        

                        moving_average_model_inference_time.get_moving_average()*1000,
                        moving_average_model_inference_time.get_moving_average() / total_time * 100,
                        
                        moving_average_img_postprocessing_time.get_moving_average()*1000,
                        moving_average_img_postprocessing_time.get_moving_average() / total_time * 100,

                        moving_average_response_signing_time.get_moving_average()*1000,
                        moving_average_response_signing_time.get_moving_average() / total_time * 100,

                        

                        moving_average_reply_time.get_moving_average() *1000,
                        moving_average_reply_time.get_moving_average() / total_time * 100,

                        
                        moving_average_image_show_time.get_moving_average()*1000,
                        moving_average_image_show_time.get_moving_average() / total_time * 100,), end='\r')

                

                        

            # counter
            image_count += 1
            #if image_count == 10000000:
            #   image_count = 0

            #cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)
                
                
                
                
                
                
            # x = time.perf_counter()
            # print(x - replied_time)
                
                
                
                # Due to the IO thread constantly fetching images, we can do any amount
                # of processing here and the next call to receive() will still give us
                # the most recent frame (more or less realtime behaviour)

                # Uncomment this statement to simulate processing load
                # limit_to_2_fps()   # Comment this statement out to run full speeed

                #cv2.imshow("Pub Sub Receive", image)
                #cv2.waitKey(1)
            
    except (KeyboardInterrupt, SystemExit):
        print('Exit due to keyboard interrupt')
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        receiver.close()
        sys.exit()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass