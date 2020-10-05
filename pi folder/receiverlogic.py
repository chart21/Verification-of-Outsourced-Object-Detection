import socket
import threading
import time
import queue
import sys
import os
from imageCounter import ImageCounter


class Receiver:

    def __init__(self, image_counter, ip, port):

        self._close = False
        self._q = queue.Queue()
        self._image_counter = image_counter
        self._ip = ip
        self._port = port
        self._connection_established = False
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def get(self):
        item = -1
        if self._q.empty():
            return item
        else:
            item = self._q.get()
            self._q.task_done()
            return item

    def getConnectionEstablished(self):
        return self._connection_established

    def getAll(self):
        item = []
        while not self._q.empty():
            item.append(self._q.get())
            self._q.task_done()
        return item

    def close(self):
        self._close = True

    def getSize(self):
        return self._q.qsize()

    def _run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # now our endpoint knows about the OTHER endpoint.
        self.s.bind((self._ip, self._port))
        self.s.listen(0)
        self.clientsocket, self.address = self.s.accept()
        self._connection_established = True
        self.headersize = 10

        msglen = 0
        new_msg = True
        while not self._close:

            if new_msg:
                msg = self.clientsocket.recv(10)
                try:
                    msglen = int(msg)
                    new_msg = False
                except:
                    # print(msg)
                    pass

            else:
                msg = self.clientsocket.recv(msglen)
                full_msg = msg.decode('latin1')
                self._q.put(full_msg)
                # if(len(full_msg) > 2) :
                self._image_counter.setOutputCounter(
                    int(full_msg[5:].split(':', 1)[0]))
                new_msg = True

                #print(self._image_counter.getNumberofOutputsReceived(), self._image_counter.getInputCounter(),full_msg[5:].split(':', 1)[0])
        self.s.close()

        
