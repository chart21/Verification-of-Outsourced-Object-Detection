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
import cv2
from nacl.signing import VerifyKey


class VideoStreamSubscriber:

    def __init__(self, hostname, port, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk, input_size):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data = ''
        self._data2 = ''
        self._data_ready = threading.Event()
        self._data2_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread2 = threading.Thread(target=self._run2, args=(merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk, input_size))
        self._thread.daemon = True
        self._thread2.daemon = True
        self._thread.start()
        self._thread2.start()


    def receive(self, timeout=15.0):
        #a = 0        
        #waited = False
        #if not self._data_ready.is_set() :
            #a = time.perf_counter()
            #waited = True       
        
            
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Contract aborted: Outsourcer at tcp://{}:{}".format(self.hostname, self.port) + 'timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review')

        #if waited :
            #print('Waited', (time.perf_counter() - a)*1000)

        self._data_ready.clear()

        
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        print('here6')
        #countera = 0
        #counterb = 0
        while not self._stop:                      
            #name, compressed = receiver.recv_jpg()

            #decompressed = cv2.imdecode(
            #np.frombuffer(compressed, dtype='uint8'), -1)
            
            #self._data = name, compressed, decompressed
            
            #countera += 1
            #print(countera)
            #f = time.perf_counter()
            #time.sleep(0.05)
            #counterb += 1
            
            #print(counterb, time.perf_counter() - f)
            #self._data_ready.set()
            self._data = receiver.recv_jpg()
            self._data_ready.set()
            


        receiver.close()


    def _run2(self, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk, input_size):
        while not self._stop:  
            name, compressed = self.receive()

            decompressedImage = cv2.imdecode(
                np.frombuffer(compressed, dtype='uint8'), -1)


            #self._data2 = name, compressed, decompressed

            if name == 'abort':
                sys.exit('Contract aborted by outsourcer according to custom')
                
            if merkle_tree_interval == 0:
                try:
                    vk.verify(bytes(compressed) + contractHash +
                            bytes(name[-2]) + bytes(name[-1]), bytes(name[:-2]))
                except:
                    sys.exit(
                        'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review')
                # print(vrification_result)

               # if name[-1] < (image_count-2)*minimum_receive_rate_from_contractor:
               #     sys.exit(
               #         'Contract aborted: Outsourcer did not acknowledge enough ouputs. Possible Consquences for Outsourcer: Blacklist, Bad Review')

            else:
                # verify if signature matches image, contract hash, and image count, and number of intervals, and random number
                try:
                    vk.verify(bytes(compressed) + contractHash +
                            bytes(name[-5]) + bytes(name[-4]) + bytes(name[-3]) + bytes(name[-2]) + bytes(name[-1]),  bytes(name[:-5]))
                except:
                    sys.exit(
                        'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review')

               # if name[-4] < (image_count-2)*minimum_receive_rate_from_contractor:
               #     sys.exit(
               #         'Contract aborted: Outsourcer did not acknowledge enough ouputs. Possible Consquences for Outsourcer: Blacklist, Bad Review')





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


            self._data2 = (images_data, name, original_image)

            self._data2_ready.set()



    def receive2(self, timeout=15.0):
        flag = self._data2_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Contract aborted11: Outsourcer at tcp://{}:{}".format(self.hostname, self.port) + 'timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review')

        #if waited :
            #print('Waited', (time.perf_counter() - a)*1000)

        self._data2_ready.clear()

        return self._data2


    def close(self):
        self._stop = True

# Simulating heavy processing load
def limit_to_2_fps():
    sleep(0.5)