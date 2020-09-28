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
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    vk = VerifyKey(OutsourceContract.public_key_outsourcer)
    sk = SigningKey(Parameters.private_key_contractor)

    model = OutsourceContract.model
    framework = Parameters.framework
    tiny = OutsourceContract.tiny
    weights = Parameters.weights
    count = Parameters.count
    dont_show = Parameters.dont_show
    info = Parameters.info
    crop = Parameters.crop
    input_size = Parameters.input_size
    iou = Parameters.iou
    score = Parameters.score
    merkle_tree_interval = OutsourceContract.merkle_tree_interval
    hostname = Parameters.ip_outsourcer  # Use to receive from other computer
    port = Parameters.port_outsourcer
    sendingPort = Parameters.sendingPort

    contractHash = Helperfunctions.hashContract().encode('latin1')
    print(contractHash)

    # configure gpu usage
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # configure video stream receiver
    receiver = vss.VideoStreamSubscriber(hostname, port)
    print('RPi Stream -> Receiver Initialized')
    time.sleep(1.0)

    # configure responder
    responder = re.Responder(hostname, sendingPort)

    # load model
    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
    else:
        saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING])

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # statistics info
    moving_average_points = 50

    # statistics
    moving_average_fps = MovingAverage(moving_average_points)
    moving_average_receive_time = MovingAverage(moving_average_points)
    moving_average_decompress_time = MovingAverage(moving_average_points)

    #moving_average_model_load_image_time = MovingAverage(moving_average_points)
    moving_average_img_preprocessing_time = MovingAverage(
        moving_average_points)

    moving_average_model_inference_time = MovingAverage(moving_average_points)

    moving_average_img_postprocessing_time = MovingAverage(
        moving_average_points)

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

    while True:

        start_time = time.perf_counter()

        # receive image
        # name[:-2] image signature, name
        name, compressed = receiver.receive()
        received_time = time.perf_counter()

        # decompress image
        decompressedImage = cv2.imdecode(
            np.frombuffer(compressed, dtype='uint8'), -1)

        decompressed_time = time.perf_counter()

       # print(name[-1])

        # verify image  (verify if signature matches image, contract hash and image count )
        try:
            vk.verify(bytes(compressed) + contractHash + bytes(name[-1]), bytes(name[:-1]))
        except:
            sys.exit('Contract aborted: Outsourcer singature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review')
        # print(vrification_result)
        verify_time = time.perf_counter()

        # image preprocessing

        original_image = cv2.cvtColor(decompressedImage, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(
            original_image, (input_size, input_size))  # 0.4ms

        image_data = image_data / 255.  # 2.53ms

        images_data = []

        for i in range(1):
            images_data.append(image_data)

        images_data = np.asarray(images_data).astype(np.float32)  # 3.15ms

        image_preprocessing_time = time.perf_counter()

        # inference

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

        model_inferenced_time = time.perf_counter()

        # image postprocessing

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
                original_image, pred_bbox, name[-1], info, counted_classes, allowed_classes=allowed_classes)
        else:
            boxtext, image = utils.draw_bbox(
                original_image, pred_bbox, name[-1], info, allowed_classes=allowed_classes)  # 0.5ms

        image = Image.fromarray(image.astype(np.uint8))  # 0.3ms

        image_postprocessing_time = time.perf_counter()

        # sign message ->need to add image_count/interval_count (for merkle tree sig), contract hash to output and verificaton

        if merkle_tree_interval == 0:
            #sig = sk.sign_deterministic(boxtext.encode('latin1'))
            sig = sk.sign(boxtext.encode('latin1') + contractHash).signature
            #sig = list(sig)
            sig = sig.decode('latin1')

            # send reply

            responder.respond(boxtext + ';--' + sig)

        else:
            mt.add_leaf(boxtext, True)
            response = boxtext
            if image_count > 1 and (image_count+1) % merkle_tree_interval == 0:
                mt.make_tree()
                merkle_root = mt.get_merkle_root()
                #sig = sk.sign_deterministic(merkle_root.encode('latin1'))
                sig = sk.sign(merkle_root.encode('latin1') + bytes(interval_count) + contractHash).signature
                response += ';--' + str(merkle_root) + \
                    ';--' + sig.decode('latin1')
                interval_count += 1
                mt = MerkleTools()
            responder.respond(response)

        response_signing_time = time.perf_counter()

        replied_time = time.perf_counter()

        # display image

        if not dont_show:
            # image.show()

            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imshow('raspberrypi', image)

            if cv2.waitKey(1) == ord('q'):
                sys.exit('Contract aborted: Contractor ended contract according to custom')

        image_showed_time = time.perf_counter()

        # statistics

        moving_average_fps.add(1 / (image_showed_time - start_time))

        moving_average_receive_time.add(received_time - start_time)

        moving_average_decompress_time.add(decompressed_time - received_time)

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
            a = time.perf_counter() - a
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
    app.run(main)