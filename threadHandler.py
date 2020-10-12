#This class initilazises 3 threads to deal with taks in paralllel to CNN inference
import sys
import socket
import traceback
import cv2
from imutils.video import VideoStream
import imagezmq
import threading
import numpy as np
import time
import cv2
from nacl.signing import VerifyKey
from nacl.signing import SigningKey
from parameters import Parameters

import Responder as re
from merkletools import MerkleTools

import json


class ThreadHandler:

    def __init__(self, hostname, port, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk_Bytes, input_size, sendingPort):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._stop_message = ''
        self._data = ''
        self._data2 = ''
        self._data3 = ''
        self._data_ready = threading.Event()
        self._data2_ready = threading.Event()
        self._image_count = 0

        self._received = threading.Event()
        self._readyToReceive = threading.Event()

        self._thread2 = threading.Thread(target=self._run, args=())
        self._thread3 = threading.Thread(target=self._run2, args=(
            merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk_Bytes, input_size))
        self._thread4 = threading.Thread(target=self._run3, args=(
            merkle_tree_interval, contractHash, hostname, sendingPort))
        self._thread2.daemon = True
        self._thread3.daemon = True
        self._thread4.daemon = True
        self._thread2.start()
        self._thread3.start()
        self._thread4.start()



    # Thread 2 (receiving) section

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            if self._stop:
                    sys.exit()
            else:
                    self._stop = True
                    self._stop_message =  "Contract aborted in Thread2 waiting for new images: Outsourcer timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review"
                    print(self._stop_message)
                    raise TimeoutError(
                    "Contract aborted in Thread2 waiting for new images: Outsourcer timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review")

 
        self._data_ready.clear()

        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub(
            "tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
 
        while not self._stop:

            self._data = receiver.recv_jpg()
            self._data_ready.set()

        receiver.close()


    # Thread 3 (decompressing, verifying, preporcessing) section

    def _run2(self, merkle_tree_interval, contractHash, minimum_receive_rate_from_contractor, vk_Bytes, input_size):
        vk = VerifyKey(vk_Bytes)
        while not self._stop:
            name, compressed = self.receive()



            decompressedImage = cv2.imdecode(
                np.frombuffer(compressed, dtype='uint8'), -1)



            

            if name == 'abort':
                if self._stop:
                    sys.exit(self._stop_message)
                else:
                    self._stop = True
                    self._stop_message = 'Contract aborted by outsourcer according to custom'
                    print(self._stop_message)
                    sys.exit(self._stop_message)

            if merkle_tree_interval == 0:
                try:
                    vk.verify(bytes(compressed) + contractHash +
                              bytes(name[-2]) + bytes(name[-1]), bytes(name[:-2]))
                except:
                    if self._stop:
                        sys.exit(self._stop_message)
                    else:
                        self._stop = True
                        self._stop_message = 'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review'
                        print(self._stop_message)
                        sys.exit(self._stop_message)
                

                if name[-1] < (self._image_count-2)*minimum_receive_rate_from_contractor:
                    sys.exit(
                        'Contract aborted: Outsourcer did not acknowledge enough ouputs. Possible Consquences for Outsourcer: Blacklist, Bad Review')

            else:
                # verify if signature matches image, contract hash, and image count, and number of intervals, and random number
                try:
                    vk.verify(bytes(compressed) + contractHash +
                              bytes(name[-5]) + bytes(name[-4]) + bytes(name[-3]) + bytes(name[-2]) + bytes(name[-1]),  bytes(name[:-5]))
                except:
                    if self._stop:
                        sys.exit(self._stop_message)
                    else:
                        self._stop = True
                        self._stop_message =  'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review' 
                        print(self._stop_message)
                        sys.exit(self._stop_message)

                if name[-4] < (self._image_count-2)*minimum_receive_rate_from_contractor:
                    sys.exit(
                        'Contract aborted: Outsourcer did not acknowledge enough ouputs. Possible Consquences for Outsourcer: Blacklist, Bad Review')

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
            if self._stop:
                sys.exit()
            else:
                if self._stop:
                    sys.exit(self._stop_message)
                else:
                    self._stop = True
                    self._stop_message = "Contract aborted in Thread3 receving from Thread2: Outsourcer timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review"  
                    print(self._stop_message)        
                    raise TimeoutError(
                        "Contract aborted in Thread3 receving from Thread2: Outsourcer timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review")

    

        self._data2_ready.clear()

        return self._data2

    # Thread 4 (signing, sending, displaying) section

    def putData(self, data, timeout=15):        
        flag = self._readyToReceive.wait(timeout=timeout)
        if not flag:
            if self._stop:
                sys.exit(self._stop_message)
            else:
                self._stop = True
                self._stop_message = "Contract aborted in Thread1 waiting for Thread4: Outsourcer probably timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review"
                print(self._stop_message)
                raise TimeoutError(
                    "Contract aborted in Thread3 receving from Thread2: Outsourcer timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review")
        self._readyToReceive.clear()

  

        self._data3 = data
 
        self._received.set()

    def _run3(self, merkle_tree_interval, contractHash, hostname, sendingPort):
        self._readyToReceive.set()
        
        sk = SigningKey(Parameters.private_key_self)
        dont_show = Parameters.dont_show

        if Parameters.is_contractor == True:
            display_name = 'Contractor'
        else:
            display_name = 'Verifier'

    

        responder = re.Responder(hostname, sendingPort)

        if merkle_tree_interval > 0:
            mt = MerkleTools()
            mtOld = MerkleTools()
            interval_count = 0
            mtOld_leaf_indices = {}
            mt_leaf_indices = {}
      
            current_challenge = 1
            merkle_root = ''
           
            last_challenge = 0

        while not self._stop:
            
            self._received.wait()
            self._received.clear()

            boxtext = self._data3[0]
            image = self._data3[1]
            name = self._data3[2]
            self._image_count = self._data3[3]
            
            if merkle_tree_interval == 0:
                    
                    sig = sk.sign(boxtext.encode('latin1') +
                                  contractHash).signature
                    
                    sig = sig.decode('latin1')

                    # send reply

                    responder.respond(boxtext + ';--' + sig)

            else:
                image_count = self._image_count
                outsorucer_signature = name[:-5]
                outsourcer_image_count = name[-5]
                outsourcer_number_of_outputs_received = name[-4]
                outsourcer_random_number = name[-3]
                outsourcer_interval_count = name[-2]
                outsourcer_time_to_challenge = bool(name[-1])      
             
                mt.add_leaf(boxtext, True)
                    
                mt_leaf_indices[outsourcer_image_count] = image_count % merkle_tree_interval
                    

                response = boxtext

                    # time to send a new merkle root
                    # e.g. if inervall = 128 then all respones from 0-127 are added to the merkle tree
                if image_count > 1 and (image_count+1) % merkle_tree_interval == 0:
              
                    mt.make_tree()
                    merkle_root = mt.get_merkle_root()

                    sig = sk.sign(merkle_root.encode(
                            'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                        # resond with merkle root
                    response += ';--' + str(merkle_root) + \
                        ';--' + sig.decode('latin1')

                    interval_count += 1
                    mtOld = mt  # save old merkle tree for challenge
                       
                    mtOld_leaf_indices.clear()
                    mtOld_leaf_indices = mt_leaf_indices.copy()  # save old indices for challenge
                      
                    mt_leaf_indices.clear()  # clear for new indices
                        

                    mt = MerkleTools()  # construct new merkle tree for next interval
                  

                else:
                        # if this is true then the outsourcer has not received the merkle root yet -> send again
                    if interval_count > outsourcer_image_count:

                        sig = sk.sign(merkle_root.encode(
                            'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                        response += ';--' + str(merkle_root) + \
                            ';--' + sig.decode('latin1')

                       

                    else:  # in this case outsourcer has confirmed to have recieved the merkle root

                            # in this case outsourcer has sent a challenge to meet with the old merkle tree, give outsourcer 3 frames time to confirm challenge received before sending again
                        if outsourcer_time_to_challenge and image_count - last_challenge > 3:
                            last_challenge = image_count
                            if outsourcer_random_number in mtOld_leaf_indices:
                                    # if challenge can be found, send proof back
                                outsourcer_random_number_index = mtOld_leaf_indices[
                                    outsourcer_random_number]

                            else:
                                    # if challenge index cannot be found return leaf 0
                                outsourcer_random_number_index = 0
                                  

                            proofs = mtOld.get_proof(
                                    outsourcer_random_number_index)

                            stringsend = ''
                            for proof in proofs:
                                stringsend += ';--'  # indicate start of proof
                                stringsend += proof.__str__()  # send proof

                            stringsend += ';--'
                                # send leaf
                            stringsend += mtOld.get_leaf(
                                outsourcer_random_number_index)
                            stringsend += ';--'
                            stringsend += mtOld.get_merkle_root()  # send root

                            stringarr = []
                            stringarr = stringsend.split(';--')

                            leaf_node = stringarr[-2]
                            root_node = stringarr[-1]
                            proof_string = stringarr[0:-2]

                            sig = sk.sign(str(stringarr[1:]).encode('latin1') + bytes(
                                interval_count-1) + contractHash).signature  # sign proof and contract details
                              
                                # attach signature
                            response += ';--' + sig.decode('latin1')
                            response += stringsend  # attach challenge response to response

                           

                responder.respond(response)

    
                # display image

            if not dont_show:
                    # image.show()

                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                cv2.imshow(display_name, image)

                if cv2.waitKey(1) == ord('q'):
                    responder.respond('abort12345:6')                    
                    if self._stop:
                        sys.exit(self._stop_message)
                    else:
                        self._stop = True
                        self._stop_message = 'Contract aborted: Contractor ended contract according to custom'
                        print(self._stop_message)
                        sys.exit(self._stop_message)

           

            self._readyToReceive.set()
                

    def close(self):
        self._stop = True


# Simulating heavy processing load

def limit_to_2_fps():
    sleep(0.5)
