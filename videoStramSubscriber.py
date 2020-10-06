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


class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

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
            self._data = receiver.recv_jpg()
            
            
            #countera += 1
            #print(countera)
            #f = time.perf_counter()
            #time.sleep(0.05)
            #counterb += 1
            
            #print(counterb, time.perf_counter() - f)
            self._data_ready.set()
            


        receiver.close()

    def close(self):
        self._stop = True

# Simulating heavy processing load
def limit_to_2_fps():
    sleep(0.5)