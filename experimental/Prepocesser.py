import sys

import socket
import traceback
import cv2
from imutils.video import VideoStream
import imagezmq
import threading
import numpy as np
#from time import sleep
import time

import videoStramSubscriber as vss
import cv2
import parameters
import sys
from nacl.signing import VerifyKey


class Preprocesser:

    def __init__(self, receiver, vk, merkle_tree_interval, minimum_receive_rate_from_contractor):
        self.receiver = receiver
        
        
        self._stop = False
        self._data_ready2 = threading.Event()
        self._thread2 = threading.Thread(target=self._run2, args=(vk, merkle_tree_interval, minimum_receive_rate_from_contractor))
        self._thread2.daemon = True
        self._thread2.start()

    def receive(self, timeout=45.0):
        #a = 0        
        #waited = False
        #if not self._data_ready.is_set() :
            #a = time.perf_counter()
            #waited = True       
        
            
        flag = self._data_ready2.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Contract aborted: Outsourcer at tcp://{}:{}".format(self.hostname, self.port) + 'timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review')

        #if waited :
            #print('Waited', (time.perf_counter() - a)*1000)

        self._data_ready2.clear()

        
        return self._data

    def _run2(self, vk, merkle_tree_interval, minimum_receive_rate_from_contractor):
        #receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        #countera = 0
        #counterb = 0
        while not self._stop:                      
            #self._data = receiver.receive()

            print('here')
            name, compressed = self.receiver.receive()
            print('here2')

            if name == 'abort':
                sys.exit('Contract aborted by outsourcer according to custom')


            received_time = time.perf_counter()

            # decompress image
            decompressedImage = cv2.imdecode(
                np.frombuffer(compressed, dtype='uint8'), -1)

            # endregion

            decompressed_time = time.perf_counter()

            # verify image  (verify if signature matches image, contract hash and image count, and number of outptuts received)
            if merkle_tree_interval == 0:
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

                print('here2')

        
        
            #print(name[-2], image_count, name[-3])

            verify_time = time.perf_counter()

            # image preprocessing

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

            image_preprocessing_time = time.perf_counter()

            print('here3')

            self._data = (images_data, name)
            
            
            #countera += 1
            #print(countera)
            #f = time.perf_counter()
            #time.sleep(0.05)
            #counterb += 1
            
            #print(counterb, time.perf_counter() - f)
            self._data_ready2.set()
            


        receiver.close()

    def close(self):
        self._stop = True

# Simulating heavy processing load
def limit_to_2_fps():
    sleep(0.5)