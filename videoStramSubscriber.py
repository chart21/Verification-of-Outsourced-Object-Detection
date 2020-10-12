#This class always fetches the newest available frame from the socket in a separate thread
# By consuming all frames it prevents of frames being piled up in a queue at the socket

import sys
import socket
import traceback
import cv2
from imutils.video import VideoStream
import imagezmq
import threading
import numpy as np
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

        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Contract aborted: Outsourcer at tcp://{}:{}".format(self.hostname, self.port) + 'timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review')

        self._data_ready.clear()

        
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
 
        while not self._stop:                      
            self._data = receiver.recv_jpg()
  
            self._data_ready.set()
            


        receiver.close()

    def close(self):
        self._stop = True

# Simulating heavy processing load
def limit_to_2_fps():
    sleep(0.5)