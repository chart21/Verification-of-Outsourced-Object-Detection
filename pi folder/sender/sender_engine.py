import time

import cv2
import imagezmq

import hmac
import hashlib

from ecdsa import SigningKey


class Sender:
    

    """
    Sender class to do actual sending related process
    """
    __slots__ = 'target_ip', 'target_port', 'sender', 'quality', 'pk'

    def __init__(self, target_ip, target_port, pk):
        self.target_ip = target_ip
        self.target_port = target_port
        self.pk = pk
        self.quality = 95
        

        # imagezmq backend
        self.sender = imagezmq.ImageSender("tcp://*:{}".format(target_port), REQ_REP=False)
        

    def set_quality(self, quality):
        """
        Image compressing quality.
        The lower, the better networking efficiency.
        Lower image quality may impact deep learning quality.
        :param quality: Image quality.
        :return: nothing
        """
        self.quality = quality

    def send_image_raw(self, name, image):
        """
        Send raw image (numpy array), low efficiency.
        :param name: Name.
        :param image: Image input as numpy array.
        :return: nothing
        """
        self.sender.send_image(name, image)

    def send_image_compressed(self, name, image):
        """
        Send compressed image (jpg), high efficiency
        :param name: Name.
        :param image: Image input as numpy array.
        :return: statistics of how long time it takes to compress, and to send image
        """
        start_time = time.perf_counter()

        # compress image
        _, compressed_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        compress_finish_time = time.perf_counter()
        
        
        #secret_key = b"NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
        #message = signature = hmac.new(secret_key, compressed_image, hashlib.sha256).hexdigest()
        #print("signature = {0}".format(signature))
        
        #ECDSA
        #message = str(self.pk.sign(compressed_image))
        #message = self.pk.sign(compressed_image)
        message = signature = self.pk.sign_deterministic(compressed_image)
        intMessage = list(message)
        #name = intMessage
        intMessage.append(name) #append image count
        
        sign_image_time = time.perf_counter()

        # send image
        #boundingBoxes = self.sender.send_jpg(name, compressed_image)
        self.sender.send_jpg(intMessage, compressed_image)
            
        send_time = time.perf_counter()

        return compress_finish_time - start_time, sign_image_time - compress_finish_time,  send_time - sign_image_time
