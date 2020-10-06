"""pub_sub_receive.py -- receive OpenCV stream using PUB SUB."""


from parameters import ParticipantData
from parameters import Parameters
from parameters import OutsourceContract
from parameters import Helperfunctions
import json
from merkletools import MerkleTools
import sys
import videoStramSubscriber as vss
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import time
import imagezmq
import Responder as re
from utilities.stats import MovingAverage

import numpy as np
import cv2
from PIL import Image


# from absl.flags import FLAGS
# from absl import app, flags, logging

import os

import multiprocessing as mp

import Datagetter2

import threading as mtd



import cv2



# comment out below line to enable tensorflow outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def inference(preprocess_queue, inference_queue):


    
    


    import tensorflow as tf
    import core.utils as utils
    
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.compat.v1 import InteractiveSession
    from tensorflow.compat.v1 import ConfigProto
    from core.functions import count_objects, crop_objects 
    from core.config import cfg
    from core.utils import read_class_names
    import os
    import random
    from core.yolov4 import filter_boxes

    tf.keras.backend.clear_session()


    
    input_size = Parameters.input_size


    model = OutsourceContract.model
    framework = Parameters.framework
    tiny = OutsourceContract.tiny
    weights = Parameters.weights
    iou = Parameters.iou
    score = Parameters.score

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    try:
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

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

    count = Parameters.count
    info = Parameters.info
    crop = Parameters.crop

    while True:
        if not preprocess_queue.empty():
            queueData = preprocess_queue.get()
            while not preprocess_queue.empty():
                queueData = preprocess_queue.get()
            #preprocess_queue.task_done()
            images_data = queueData[0]
            name = queueData[1]
            original_image = queueData[2]

            #preprocess_queue.task_done()

            if framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(
                    output_details[i]['index']) for i in range(len(output_details))]
                if model == 'yolov3' and tiny == True:
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


            boxes, scores, classes, valid_detections=tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )  # 1.2ms


            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            
            original_h, original_w, _=original_image.shape

            bboxes=utils.format_boxes(
                boxes.numpy()[0], original_h, original_w)  # 1ms #-> no tf needed

            # hold all detection data in one variable
            pred_bbox=[bboxes, scores.numpy()[0], classes.numpy()[0],
                        valid_detections.numpy()[0]]

            # by default allow all classes in .names file
            allowed_classes=list(class_names.values())

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']

            # if crop flag is enabled, crop each detection and save it as new image
            if crop:
                crop_path=os.path.join(
                    os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                            pred_bbox, crop_path, allowed_classes)

            if count:
                # count objects found
                counted_classes=count_objects(
                    pred_bbox, by_class=False, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                boxtext, image=utils.draw_bbox(
                    original_image, pred_bbox, info, counted_classes, allowed_classes=allowed_classes)
            else:
                boxtext, image=utils.draw_bbox(
                    original_image, pred_bbox, info, allowed_classes=allowed_classes)  # 0.5ms

            image=Image.fromarray(image.astype(np.uint8))  # 0.3ms


            inference_queue.put((boxtext, image, name))


def preprocessing(datagetter):
    
    

    vk = VerifyKey(OutsourceContract.public_key_outsourcer)

    input_size = Parameters.input_size
    merkle_tree_interval = OutsourceContract.merkle_tree_interval

    hostname = Parameters.ip_outsourcer  # Use to receive from other computer
    port = Parameters.port_outsourcer

    minimum_receive_rate_from_contractor = Parameters.minimum_receive_rate_from_contractor

    contractHash = Helperfunctions.hashContract().encode('latin1')

    # configure video stream receiver
    receiver = vss.VideoStreamSubscriber(hostname, port)
    print('RPi Stream -> Receiver Initialized')
    old_imagecount = -1
    while True:
        st = time.perf_counter()
        name, compressed = receiver.receive()
        if name == 'abort':
            sys.exit('Contract aborted by outsourcer according to custom')

        if (merkle_tree_interval == 0 and name[-1] != old_imagecount) or (merkle_tree_interval > 0 and name[-5] != old_imagecount):

            decompressedImage = cv2.imdecode(
                np.frombuffer(compressed, dtype='uint8'), -1)

            if merkle_tree_interval == 0:
                old_imagecount = name[-1]

                try:
                    vk.verify(bytes(compressed) + contractHash +
                                    bytes(name[-2]) + bytes(name[-1]), bytes(name[:-2]))
                except:
                    sys.exit(
                                'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review')
                        # print(vrification_result)

                    if name[-1] < (image_count-2)*minimum_receive_rate_from_contractor:
                        sys.exit(
                                'Contract aborted: Outsourcer did not acknowledge enough ouputs. Possible Consquences for Outsourcer: Blacklist, Bad Review')

            else:
                old_imagecount = name[-5]
                        # verify if signature matches image, contract hash, and image count, and number of intervals, and random number
                try:
                    vk.verify(bytes(compressed) + contractHash +
                                bytes(name[-5]) + bytes(name[-4]) + bytes(name[-3]) + bytes(name[-2]) + bytes(name[-1]),  bytes(name[:-5]))
                except:
                    sys.exit(
                                'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review')

                if name[-4] < (image_count-2)*minimum_receive_rate_from_contractor:
                    sys.exit(
                            'Contract aborted: Outsourcer did not acknowledge enough ouputs. Possible Consquences for Outsourcer: Blacklist, Bad Review')

                outsorucer_signature = name[:-5]
                outsourcer_image_count = name[-5]
                outsourcer_number_of_outputs_received = name[-4]
                outsourcer_random_number = name[-3]
                outsourcer_interval_count = name[-2]
                outsourcer_time_to_challenge = bool(name[-1])

    # region
            original_image = cv2.cvtColor(decompressedImage, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(
                original_image, (input_size, input_size))  # 0.4ms

            image_data = image_data / 255.  # 2.53ms

            images_data = []

            for i in range(1):
                images_data.append(image_data)

            images_data = np.asarray(images_data).astype(np.float32)  # 3.15ms

            # endregion
            aq = time.perf_counter()
            print('pre',aq-st)
            #preprocess_queue.put((images_data, name, original_image))
            datagetter.setData((images_data, name, original_image))


# from object_detection.object_detection import Model
# from utilities.render import Render


# from ecdsa import VerifyingKey
# from ecdsa import SigningKey


# Helper class implementing an IO deamon thread

def dummy():
    while True:
        #print('jo')
        a = 0

def main():

    # get paramters and contract details
    

    

    
    # print(contractHash)

    #preprocess_queue = queue.LifoQueue()
    #inference_queue = queue.LifoQueue()
    preprocess_queue = mp.Queue()
    #inference_queue = mp.Queue()

    # postprocess_queue = Queue()
    dg = Datagetter2.Datagetter2()

    #p1 = mp.Process(target=inference, args= (preprocess_queue, inference_queue))
    #p2 = mp.Process(target=preprocessing, args=(dg,))
    p2 = mp.Process(target=preprocessing, args=(dg,))
    #p1 = Process(target=dummy)
    #p2 = Process(target=dummy)

    # p3 = Process(target=Show_Image_mp, args=(Processed_frames, show, Final_frames))
    #p1.start()
    p2.start()
    # p3.start()

    sk = SigningKey(Parameters.private_key_contractor)

    contractHash = Helperfunctions.hashContract().encode('latin1')

    

    dont_show = Parameters.dont_show

    

    merkle_tree_interval = OutsourceContract.merkle_tree_interval
    hostname = Parameters.ip_outsourcer  # Use to receive from other computer
    port = Parameters.port_outsourcer
    sendingPort = Parameters.sendingPort


    #import tensorflow as tf



    # time.sleep(1.0)






    # configure responder
    responder=re.Responder(hostname, sendingPort)

    # statistics info
    moving_average_points=50

    # statistics
    moving_average_fps=MovingAverage(moving_average_points)
    moving_average_receive_time=MovingAverage(moving_average_points)
    moving_average_decompress_time=MovingAverage(moving_average_points)

    # moving_average_model_load_image_time = MovingAverage(moving_average_points)
    moving_average_img_preprocessing_time=MovingAverage(
        moving_average_points)

    moving_average_model_inference_time=MovingAverage(moving_average_points)

    moving_average_img_postprocessing_time=MovingAverage(
        moving_average_points)

    moving_average_reply_time=MovingAverage(moving_average_points)
    moving_average_image_show_time=MovingAverage(moving_average_points)
    moving_average_verify_image_sig_time=MovingAverage(moving_average_points)

    moving_average_response_signing_time=MovingAverage(moving_average_points)

    image_count=0

    a=0
    b=0

    if merkle_tree_interval > 0:
        mt=MerkleTools()
        mtOld=MerkleTools()
        interval_count=0
        mtOld_leaf_indices={}
        mt_leaf_indices={}
        # rendundancy_counter = 0
        # rendundancy_counter2 = 0
        current_challenge=1
        merkle_root=''
        # stringsend = ''
        last_challenge=0
    image_showed_time=time.perf_counter()  # init

    import tensorflow as tf
    import core.utils as utils
    
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.compat.v1 import InteractiveSession
    from tensorflow.compat.v1 import ConfigProto
    from core.functions import count_objects, crop_objects 
    from core.config import cfg
    from core.utils import read_class_names
    import os
    import random
    from core.yolov4 import filter_boxes

    tf.keras.backend.clear_session()


    
    input_size = Parameters.input_size


    model = OutsourceContract.model
    framework = Parameters.framework
    tiny = OutsourceContract.tiny
    weights = Parameters.weights
    iou = Parameters.iou
    score = Parameters.score

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    try:
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

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

    count = Parameters.count
    info = Parameters.info
    crop = Parameters.crop

    while True:
        queueData = dg.get_data() 
        if queueData != -1:
            #queueData = preprocess_queue.get()
            #while not preprocess_queue.empty():
            #    queueData = preprocess_queue.get()
            #queueData = dg.get_data()
            a = time.perf_counter()
            #preprocess_queue.task_done()
            images_data = queueData[0]
            name = queueData[1]
            original_image = queueData[2]

            #preprocess_queue.task_done()

            if framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(
                    output_details[i]['index']) for i in range(len(output_details))]
                if model == 'yolov3' and tiny == True:
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


            boxes, scores, classes, valid_detections=tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )  # 1.2ms


            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            
            original_h, original_w, _=original_image.shape

            bboxes=utils.format_boxes(
                boxes.numpy()[0], original_h, original_w)  # 1ms #-> no tf needed

            # hold all detection data in one variable
            pred_bbox=[bboxes, scores.numpy()[0], classes.numpy()[0],
                        valid_detections.numpy()[0]]

            # by default allow all classes in .names file
            allowed_classes=list(class_names.values())

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']

            # if crop flag is enabled, crop each detection and save it as new image
            if crop:
                crop_path=os.path.join(
                    os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                            pred_bbox, crop_path, allowed_classes)

            if count:
                # count objects found
                counted_classes=count_objects(
                    pred_bbox, by_class=False, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                boxtext, image=utils.draw_bbox(
                    original_image, pred_bbox, info, counted_classes, allowed_classes=allowed_classes)
            else:
                boxtext, image=utils.draw_bbox(
                    original_image, pred_bbox, info, allowed_classes=allowed_classes)  # 0.5ms

            image=Image.fromarray(image.astype(np.uint8))  # 0.3ms


            #inference_queue.put((boxtext, image, name))

    #while True:

        # start_time = time.perf_counter()


        # if not inference_queue.empty():
        #     queueData=inference_queue.get()
        #     while not inference_queue.empty():
        #         queueData=inference_queue.get()




            start_time=image_showed_time

        # # boxes, scores, classes, valid_detections, name, original_image
            #queueData=inference_queue.get()
            #inference_queue.task_done()
            # boxes=queueData[0]
            # scores=queueData[1]
            # classes=queueData[2]
            # valid_detections=queueData[3]
            # name = queueData[4]
            # original_image = queueData[5]
            # boxtext = queueData[0]
            # image = queueData[1]
            # name = queueData[2]


            if merkle_tree_interval > 0:
                outsorucer_signature=name[:-5]
                outsourcer_image_count=name[-5]
                outsourcer_number_of_outputs_received=name[-4]
                outsourcer_random_number=name[-3]
                outsourcer_interval_count=name[-2]
                outsourcer_time_to_challenge=bool(name[-1])



            received_time = time.perf_counter()
            
            image_preprocessing_time=time.perf_counter()

            decompressed_time = time.perf_counter()

            verify_time = time.perf_counter()

            # inference

            # region


            # endregion

            model_inferenced_time=time.perf_counter()

            # image postprocessing

            # region

            h=time.perf_counter()

            

            

            # endregion

            if merkle_tree_interval == 0:
                boxtext='Image' + str(name[-2]) + ':;' + boxtext
            else:
                boxtext='Image' + str(outsourcer_image_count) + ':;' + boxtext

            image_postprocessing_time=time.perf_counter()

            # sign message ->need to add image_count/interval_count (for merkle tree sig), contract hash to output and verificaton

            if merkle_tree_interval == 0:
                # sig = sk.sign_deterministic(boxtext.encode('latin1'))
                sig=sk.sign(boxtext.encode('latin1') + contractHash).signature
                # sig = list(sig)
                sig=sig.decode('latin1')

                # send reply

                responder.respond(boxtext + ';--' + sig)

            else:
                # print(image_count)
                # add leafs dynamiclly to merkle tree
                mt.add_leaf(boxtext, True)
                # remember indices for challenge
                mt_leaf_indices[outsourcer_image_count]=image_count % merkle_tree_interval
                # print(image_count % merkle_tree_interval)


                response=boxtext

                # time to send a new merkle root
                # e.g. if inervall = 128 then all respones from 0-127 are added to the merkle tree
                if image_count > 1 and (image_count+1) % merkle_tree_interval == 0:
                    # print(image_count)
                    a=time.perf_counter()
                    # rendundancy_counter = 2
                    mt.make_tree()
                    merkle_root=mt.get_merkle_root()

                    sig=sk.sign(merkle_root.encode(
                        'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                    # resond with merkle root
                    response += ';--' + str(merkle_root) + \
                        ';--' + sig.decode('latin1')

                    interval_count += 1
                    mtOld=mt  # save old merkle tree for challenge
                    # mtOld_leaf_indices.clear() # clear old indices
                    mtOld_leaf_indices.clear()
                    mtOld_leaf_indices=mt_leaf_indices.copy()  # save old indices for challenge
                    # print(mtOld_leaf_indices)
                    mt_leaf_indices.clear()  # clear for new indices
                    # mt_leaf_indices = {}

                    mt=MerkleTools()  # construct new merkle tree for next interval
                    te=time.perf_counter()-a
                # print('1', te, image_count)

                else:
                    # if this is true then the outsourcer has not received the merkle root yet -> send again
                    if interval_count > outsourcer_image_count:

                        sig=sk.sign(merkle_root.encode(
                        'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                        response += ';--' + str(merkle_root) + \
                        ';--' + sig.decode('latin1')

                    # print('2', image_count)

                    else:  # in this case outsourcer has confirmed to have recieved the merkle root

                        # in this case outsourcer has sent a challenge to meet with the old merkle tree, give outsourcer 3 frames time to confirm challenge received before sending again
                        if outsourcer_time_to_challenge and image_count - last_challenge > 3:
                            last_challenge=image_count
                            if outsourcer_random_number in mtOld_leaf_indices:
                                # if challenge can be found, send proof back
                                outsourcer_random_number_index=mtOld_leaf_indices[outsourcer_random_number]

                            else:
                                # if challenge index cannot be found return leaf 0
                                outsourcer_random_number_index=0
                                # print('proof index not found')








                            proofs=mtOld.get_proof(
                                outsourcer_random_number_index)

                            stringsend=''
                            for proof in proofs:
                                stringsend += ';--'  # indicate start of proof
                                stringsend += proof.__str__()  # send proof

                            stringsend += ';--'
                            # send leaf
                            stringsend += mtOld.get_leaf(
                                outsourcer_random_number_index)
                            stringsend += ';--'
                            stringsend += mtOld.get_merkle_root()  # send root

                            stringarr=[]
                            stringarr=stringsend.split(';--')

                            leaf_node=stringarr[-2]
                            root_node=stringarr[-1]
                            proof_string=stringarr[0:-2]

                            # sign proof and contract details
                            sig=sk.sign(str(stringarr[1:]).encode(
                                'latin1') + bytes(interval_count-1) + contractHash).signature
                            # print(str(stringarr).encode('latin1') + bytes(interval_count-1) + contractHash)
                            # print(stringarr)
                                # attach signature
                            response += ';--' + sig.decode('latin1')
                            response += stringsend  # attach challenge response to response






                        # print('3', te, image_count)


                responder.respond(response)

            response_signing_time=time.perf_counter()

        # print(response_signing_time- image_postprocessing_time)

            replied_time=time.perf_counter()

            # display image

            if not dont_show:
                # image.show()

                image=cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                cv2.imshow('raspberrypi', image)

                if cv2.waitKey(1) == ord('q'):
                    responder.respond('abort12345:6')
                    sys.exit(
                        'Contract aborted: Contractor ended contract according to custom')

            image_showed_time=time.perf_counter()

            b = time.perf_counter()
            print('inf', b-a)

            # statistics

            moving_average_fps.add(1 / (image_showed_time - start_time))

            moving_average_receive_time.add(received_time - start_time)

            moving_average_decompress_time.add(
                decompressed_time - received_time)

            moving_average_verify_image_sig_time.add(
                verify_time - decompressed_time)

            moving_average_img_preprocessing_time.add(
                image_preprocessing_time - verify_time)

            moving_average_model_inference_time.add(
                model_inferenced_time - image_preprocessing_time)

            moving_average_img_postprocessing_time.add(
                image_postprocessing_time - model_inferenced_time)

            moving_average_response_signing_time.add(
                response_signing_time - image_postprocessing_time)  # adjust for merkle root

            moving_average_reply_time.add(replied_time - response_signing_time)

            moving_average_image_show_time.add(
                image_showed_time - replied_time)

            total_time=moving_average_receive_time.get_moving_average() \
                + moving_average_decompress_time.get_moving_average() \
                + moving_average_verify_image_sig_time.get_moving_average() \
                + moving_average_img_preprocessing_time.get_moving_average() \
                + moving_average_model_inference_time.get_moving_average() \
                + moving_average_img_postprocessing_time.get_moving_average() \
                + moving_average_response_signing_time.get_moving_average() \
                + moving_average_reply_time.get_moving_average() \
                + moving_average_image_show_time.get_moving_average()

            if(image_count == 800):
                a=time.perf_counter()
            if(image_count == 1200):
                a=time.perf_counter() - a
                print(a)

            # terminal prints
            if image_count % 20 == 0:

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



                        moving_average_reply_time.get_moving_average() * 1000,
                        moving_average_reply_time.get_moving_average() / total_time * 100,


                        moving_average_image_show_time.get_moving_average()*1000,
                        moving_average_image_show_time.get_moving_average() / total_time * 100,), end='\r')

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
    #app.run(main)
    main()
