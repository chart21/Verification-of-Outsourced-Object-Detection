import time

from sender.rpi_camera import RPiCamera
from sender.sender_engine import Sender
from utilities.stats import MovingAverage



from nacl.signing import SigningKey
from nacl.signing import VerifyKey

#from ecdsa import SigningKey

#from ecdsa import VerifyingKey

from imageCounter import ImageCounter

import receiverlogic

from merkletools import MerkleTools
import json

from parameters import Helperfunctions
from parameters import OutsourceContract
from parameters import Parameters
from parameters import ParticipantData
from parameters import VerifierContract

import sys

def main():
    """
    main function interface
    :return: nothing
    """
    


    pk = SigningKey(Parameters.private_key_outsourcer)

    #print(pk.verify_key)

    vk = VerifyKey(OutsourceContract.public_key_contractor)

   
    

    # video info
    width = Parameters.input_size
    height = Parameters.input_size
    quality = Parameters.quality

    
    receiver_ip = Parameters.receiver_ip
    receiver_port = Parameters.receiver_port


    sending_port = Parameters.sending_port

    # statistics info
    moving_average_points = Parameters.moving_average_points

    

    merkle_tree_interval = OutsourceContract.merkle_tree_interval

    image_counter = ImageCounter()
    r = receiverlogic.Receiver(image_counter, receiver_ip, receiver_port)


    a = 0

    contractHash = Helperfunctions.hashContract().encode('latin1')
    print(contractHash)

    if merkle_tree_interval > 0:
        mt = MerkleTools()
        interval_count = 0

    # initialize sender
    image_sender = Sender(sending_port, pk, quality)
    image_sender.set_quality(quality)
    print('RPi Stream -> Sender Initialized')

    # initialize RPi camera
    rpi_cam = RPiCamera(width, height)
    rpi_cam.start()
    print('RPi Stream -> Camera Started')
    time.sleep(1.0)

    # statistics
    moving_average_fps = MovingAverage(moving_average_points)
    moving_average_camera_time = MovingAverage(moving_average_points)
    moving_average_compress_time = MovingAverage(moving_average_points)
    moving_average_sign_time = MovingAverage(moving_average_points)
    moving_average_send_time = MovingAverage(moving_average_points)
    moving_average_response_time = MovingAverage(moving_average_points)
    moving_average_receive_time = MovingAverage(moving_average_points)
    moving_average_verify_time = MovingAverage(moving_average_points)
    
    # streaming
    print('RPi Stream -> Start Streaming')
    while True:
        start_time = time.perf_counter()

        # time.sleep(0.005)

        # capture image
        image = rpi_cam.get_image()

        camera_time = time.perf_counter()

        
        compress_time, sign_time, send_time, = image_sender.send_image_compressed(
            image_counter.getInputCounter(), image, contractHash)
       

        # verifying

        receive_time = time.perf_counter()

        responses = []
        
        output = r.getAll()

        if merkle_tree_interval == 0:
            
            
            

            
            for o in output:
                        
                try:
                    sig = o.split(';--')[1].encode('latin1')
                    msg = o.split(';--')[0].encode('latin1')
                except:
                    sys.exit('Contract aborted: Contractor response is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review')
                        
                try:
                    vk.verify(msg + contractHash, sig)
                except:
                    sys.exit('Contract aborted: Contractor singature does not match response. Possible Consquences for Contractor: Blacklist, Bad Review')
                responses.append(msg)

            
        else:
            
            if image_counter.getNumberofOutputsReceived() > (merkle_tree_interval) * (interval_count+2):
                   sys.exit('Contract aborted: No root hash received for current interval in time. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')


            for o in output:
                root_hash_received = False
                msg = o.split(';--')[0].encode('latin1')

                
                if image_counter.getNumberofOutputsReceived() > (merkle_tree_interval-2) * (interval_count+1):
                    try: #check if merkle root received
                        
                            root_hash = o.split(';--')[1].encode('latin1')
                            sig = o.split(';--')[2].encode('latin1')
                            root_hash_received = True
                    except:
                        pass
                if root_hash_received == True: #if root has received, verify signature                    
                    
                    root_hash_received = False
                    try:

                        match = vk.verify(root_hash + bytes(interval_count) + contractHash, sig)
                        interval_count += 1
                        print(interval_count, image_counter.getNumberofOutputsReceived())
                    except:
                        sys.exit('Contract aborted: Contractor singature of root hash is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

                            
                       


                

                responses.append(msg)


        

        verify_time = time.perf_counter()
        if(OutsourceContract.criteria == 'Atleast 2 objects detected'):
            for st in responses:
                if len(st) > 200:
                    print(st)

        frames_behind = image_counter.getFramesAhead()
        

        if(image_counter.getNumberofOutputsReceived() == 800):
            a = time.perf_counter()
        if(image_counter.getNumberofOutputsReceived() == 1200):
            a = time.perf_counter() - a
            print(a)


        # statistics
        moving_average_camera_time.add(camera_time - start_time)
        moving_average_compress_time.add(compress_time)
        moving_average_sign_time.add(sign_time)
        moving_average_send_time.add(send_time)
       
        moving_average_verify_time.add(verify_time - receive_time)
        if(frames_behind != -1):
            moving_average_response_time.add(frames_behind)

        total_time = moving_average_camera_time.get_moving_average() \
            + moving_average_compress_time.get_moving_average() \
            + moving_average_sign_time.get_moving_average() \
            + moving_average_send_time.get_moving_average()

        instant_fps = 1 / (time.perf_counter() - start_time)
        moving_average_fps.add(instant_fps)

        # terminal prints
        if image_counter.getInputCounter() % 20 == 0:
            print("total: %5.1fms (%5.1ffps) camera %4.1f (%4.1f%%) compressing %4.1f (%4.1f%%) signing %4.1f (%4.1f%%) sending %4.1f (%4.1f%%) frames ahead %4.1f verify time %4.1f (%4.1f%%) "
                  % (


                      1000/moving_average_fps.get_moving_average(),
                      moving_average_fps.get_moving_average(),

                      moving_average_camera_time.get_moving_average()*1000,
                      moving_average_camera_time.get_moving_average() / total_time * 100,

                      moving_average_compress_time.get_moving_average()*1000,
                      moving_average_compress_time.get_moving_average() / total_time * 100,

                      moving_average_sign_time.get_moving_average()*1000,
                      moving_average_sign_time.get_moving_average() / total_time * 100,

                      moving_average_send_time.get_moving_average()*1000,
                      moving_average_send_time.get_moving_average() / total_time * 100,

                      moving_average_response_time.get_moving_average(),

                      moving_average_verify_time.get_moving_average()*1000,
                      moving_average_verify_time.get_moving_average() / total_time * 100), end='\r')

        # counter
        image_counter.increaseInputCounter()


       # print(c - a)
        # a = time.perf_counter()

if __name__ == '__main__':
    main()
