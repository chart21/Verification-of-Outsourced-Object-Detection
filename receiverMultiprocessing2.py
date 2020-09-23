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
from utilities.render import Render
from utilities.stats import MovingAverage



import imagezmq
import time
from ecdsa import VerifyingKey


#flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
#flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                    'path to weights file')
#flags.DEFINE_integer('size', 416, 'resize images to')
#flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
#flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
#flags.DEFINE_string('output', './detections/', 'path to output folder')
#flags.DEFINE_float('iou', 0.45, 'iou threshold')
#flags.DEFINE_float('score', 0.50, 'score threshold')
#flags.DEFINE_boolean('count', False, 'count objects within images')
#flags.DEFINE_boolean('dont_show', False, 'dont show image output')
#flags.DEFINE_boolean('info', False, 'print info on detections')
#flags.DEFINE_boolean('crop', False, 'crop detections from images')

def main(_argv):
    vk = b'Y\xf8D\xe6o\xf9MZZh\x9e\xcb\xe0b\xb7h\xdb\\\xd7\x80\xd2S\xf5\x81\x92\xe8\x109r*U\xebT\x95\x0c\xf2\xf4(\x13%\x83\xb8\xfa;\xf04\xd3\xfb'
    vk = VerifyingKey.from_string(vk)


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

    # load model
    if framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=weights)
    else:
            saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])

    
    
    # statistics info
    moving_average_points = 50
    # initialize receiver
    image_hub = imagezmq.ImageHub()
    print('RPi Stream -> Receiver Initialized')
    time.sleep(1.0)

    # initialize render
    render = Render()
    print('RPi Stream -> Render Ready') 
    
    
    # statistics
    moving_average_fps = MovingAverage(moving_average_points)
    moving_average_receive_time = MovingAverage(moving_average_points)
    moving_average_decompress_time = MovingAverage(moving_average_points)
    moving_average_model_load_image_time = MovingAverage(moving_average_points)
    moving_average_model_inference_time = MovingAverage(moving_average_points)
    moving_average_reply_time = MovingAverage(moving_average_points)
    moving_average_image_show_time = MovingAverage(moving_average_points)
    image_count = 0
    
    

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

     # streaming
    print('RPi Stream -> Receiver Streaming')

    while True:
        start_time = time.monotonic()

        # receive image
        name, compressed = image_hub.recv_jpg()
        received_time = time.monotonic()

        # decompress image
        decompressedImage = cv2.imdecode(np.frombuffer(compressed, dtype='uint8'), -1)
        decompressed_time = time.monotonic()


        vk.verify( bytes(name), compressed)
        
        #frame = cv2.cvtColor(decompressedImage, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(frame)



   
        original_image = cv2.cvtColor(decompressedImage, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        
        # get image name by using split method
        #image_name = image_path.split('/')[-1]
        #image_name = image_name.split('.')[0]

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

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

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
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
            boxtext, image = utils.draw_bbox(original_image, pred_bbox, image_count, info, counted_classes, allowed_classes=allowed_classes)
        else:
            boxtext, image = utils.draw_bbox(original_image, pred_bbox, image_count, info, allowed_classes=allowed_classes)
        
        image = Image.fromarray(image.astype(np.uint8))


        

        #print(boxtext)
        # send reply
        if(info):
            image_hub.send_reply(boxtext)
        else:
            image_hub.send_reply('Ok')
        #stra = str(pred_bbox).encode()
        #image_hub.send_reply(stra)
        #print(stra)
        #image_hub.send_reply(str(pred_bbox).encode())
        #image_hub.send_reply(bytearray(pred_bbox))


        if not dont_show:
            #image.show()
            
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imshow('raspberrypi', image)
        image_showed_time = time.monotonic()
        if cv2.waitKey(1) == ord('q'):
            break

        # statistics
        instant_fps = 1 / (image_showed_time - start_time)
        moving_average_fps.add(instant_fps)
        receive_time = received_time - start_time
        moving_average_receive_time.add(receive_time)
        decompress_time = decompressed_time - received_time
        moving_average_decompress_time.add(decompress_time)
        #model_load_image_time = model_loaded_image_time - decompressed_time
        #moving_average_model_load_image_time.add(model_load_image_time)
        #model_inference_time = model_inferenced_time - model_loaded_image_time
        #moving_average_model_inference_time.add(model_inference_time)
        #reply_time = replied_time - model_inferenced_time
        #moving_average_reply_time.add(reply_time)
        #image_show_time = image_showed_time - replied_time
        #moving_average_image_show_time.add(image_show_time)
        #total_time = moving_average_receive_time.get_moving_average() \
        #             + moving_average_decompress_time.get_moving_average() \
        #             + moving_average_model_load_image_time.get_moving_average() \
        #             + moving_average_model_inference_time.get_moving_average() \
        #             + moving_average_reply_time.get_moving_average() \
        #             + moving_average_image_show_time.get_moving_average()

         #terminal prints
        #if image_count % 10 == 0:
            #print(moving_average_fps)
            #print(decompress_time)
            #print(" receiver's fps: %4.1f"
                  #" receiver's time components: "
                  #"receiving %4.1f%% "
                  #"decompressing %4.1f%% "
                  #"model load image %4.1f%% "
                  #"model inference %4.1f%% "
                  #"replying %4.1f%% "
                  #"image show %4.1f%%"
                  #% (moving_average_fps.get_moving_average()), end='\r')
                      #moving_average_fps.get_moving_average(),
                     #moving_average_receive_time.get_moving_average() / total_time * 100,
                     #moving_average_decompress_time.get_moving_average() / total_time * 100,
                     #moving_average_model_load_image_time.get_moving_average() / total_time * 100,
                     #moving_average_model_inference_time.get_moving_average() / total_time * 100,
                     #moving_average_reply_time.get_moving_average() / total_time * 100,
                     #moving_average_image_show_time.get_moving_average() / total_time * 100), end='\r')

                    #artifically added
                     

        # counter
        image_count += 1
        #if image_count == 10000000:
        #   image_count = 0

        #cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
