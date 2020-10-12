#class to send images over a socket
import time

import cv2
import imagezmq

import hmac
import hashlib

from nacl.signing import SigningKey


class Sender:

    """
    Sender class to do actual sending related process
    """
    __slots__ = 'target_ip', 'target_port', 'sender', 'quality', 'pk'

    def __init__(self, target_port, pk, quality):
        #self.target_ip = target_ip
        self.target_port = target_port
        self.pk = pk
        self.quality = quality

        # imagezmq backend
        self.sender = imagezmq.ImageSender(
            "tcp://*:{}".format(target_port), REQ_REP=False)

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

    def send_image_compressed_Merkle(self, image_count, image, contractHash, number_of_outputs_received, random_number, interval_count, time_to_challenge):
  
        start_time = time.perf_counter()

        # compress image
        _, compressed_image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        compress_finish_time = time.perf_counter()

        # sign contracthash, image, image_count, numbrt of outputs, radnom number
        inputs_to_be_signed = bytes(compressed_image) + contractHash + bytes(
            image_count) + bytes(number_of_outputs_received) + bytes(random_number) +bytes(interval_count) + bytes(time_to_challenge)

        message = self.pk.sign(inputs_to_be_signed).signature
        intMessage = list(message)  # [:-5]

        intMessage.append(image_count)  # append image count [-5]

        # append number of outputs received to commit to a payment [-4]
        intMessage.append(number_of_outputs_received)

        

        # append random number that is used for the challenge [-3]
        intMessage.append(random_number)

        intMessage.append(interval_count) #[-2] 

        intMessage.append(int(time_to_challenge)) #-1 

        sign_image_time = time.perf_counter()

        #print(intMessage[-2], intMessage[-3], number_of_outputs_received, image_count)

        self.sender.send_jpg(intMessage, compressed_image)

        send_time = time.perf_counter()

        return compress_finish_time - start_time, sign_image_time - compress_finish_time,  send_time - sign_image_time

    def send_image_compressed_Merkle_with_return(self, image_count, image, contractHash, number_of_outputs_received, random_number, interval_count, time_to_challenge):
    
        start_time = time.perf_counter()

        # compress image
        _, compressed_image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        compress_finish_time = time.perf_counter()

        # sign contracthash, image, image_count, numbrt of outputs, radnom number
        inputs_to_be_signed = bytes(compressed_image) + contractHash + bytes(
            image_count) + bytes(number_of_outputs_received) + bytes(random_number) +bytes(interval_count) + bytes(time_to_challenge)

        message = self.pk.sign(inputs_to_be_signed).signature
        intMessage = list(message)  # [:-5]

        intMessage.append(image_count)  # append image count [-5]

        # append number of outputs received to commit to a payment [-4]
        intMessage.append(number_of_outputs_received)

        

        # append random number that is used for the challenge [-3]
        intMessage.append(random_number)

        intMessage.append(interval_count) #[-2] 

        intMessage.append(int(time_to_challenge)) #-1 

        sign_image_time = time.perf_counter()

        

        self.sender.send_jpg(intMessage, compressed_image)

        send_time = time.perf_counter()

        return compress_finish_time - start_time, sign_image_time - compress_finish_time,  send_time - sign_image_time, compressed_image
    
    def send_image_compressed_with_input(self, image_count, image, contractHash, number_of_outputs_received, compressed_image):
    
        start_time = time.perf_counter()


        compress_finish_time = start_time

        inputs_to_be_signed = bytes(compressed_image) + contractHash + bytes(image_count) + bytes(
            number_of_outputs_received)  # sign contracthash, image, image_count

        message = self.pk.sign(inputs_to_be_signed).signature
        intMessage = list(message)
        
        intMessage.append(image_count)  # append image count

        # append number of outputs received to commit to a paymnet
        intMessage.append(number_of_outputs_received)

        sign_image_time = time.perf_counter()

        self.sender.send_jpg(intMessage, compressed_image)

        send_time = time.perf_counter()

        return compress_finish_time - start_time, sign_image_time - compress_finish_time,  send_time - sign_image_time

    def send_image_compressed(self, image_count, image, contractHash, number_of_outputs_received):
      
        start_time = time.perf_counter()

        # compress image
        _, compressed_image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        compress_finish_time = time.perf_counter()

        inputs_to_be_signed = bytes(compressed_image) + contractHash + bytes(image_count) + bytes(
            number_of_outputs_received)  # sign contracthash, image, image_count

        message = self.pk.sign(inputs_to_be_signed).signature
        intMessage = list(message)
        
        intMessage.append(image_count)  # append image count

        # append number of outputs received to commit to a paymnet
        intMessage.append(number_of_outputs_received)

        sign_image_time = time.perf_counter()

        self.sender.send_jpg(intMessage, compressed_image)

        send_time = time.perf_counter()

        return compress_finish_time - start_time, sign_image_time - compress_finish_time,  send_time - sign_image_time

    def send_image_compressed_with_return(self, image_count, image, contractHash, number_of_outputs_received):
     
        start_time = time.perf_counter()

        # compress image
        _, compressed_image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        compress_finish_time = time.perf_counter()

        inputs_to_be_signed = bytes(compressed_image) + contractHash + bytes(image_count) + bytes(
            number_of_outputs_received)  # sign contracthash, image, image_count

        message = self.pk.sign(inputs_to_be_signed).signature
        intMessage = list(message)
        
        intMessage.append(image_count)  # append image count

        # append number of outputs received to commit to a paymnet
        intMessage.append(number_of_outputs_received)

        sign_image_time = time.perf_counter()

        self.sender.send_jpg(intMessage, compressed_image)

        send_time = time.perf_counter()

        return compress_finish_time - start_time, sign_image_time - compress_finish_time,  send_time - sign_image_time, compressed_image

    def send_abort(self, image):
        _, compressed_image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])        
        self.sender.send_jpg('abort', image)

