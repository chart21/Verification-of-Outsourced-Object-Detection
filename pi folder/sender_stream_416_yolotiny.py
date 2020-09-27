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


def main():
    """
    main function interface
    :return: nothing
    """
    # pk = SigningKey.generate()
    # privateKey = b'\x08\x8dv\xb3vB;\xb2\xb3\xca9\x94\x89f\xd6\xa9\x8d\x95\x10\x91\x12M\xadf'
    #privateKey = b'XN\xd3\xa6\\9\x98>P0S<\xbf\xcd\x93\xd1\x17\xb6\xf7&\xb0\xe9d\xad'
    #pk = SigningKey.from_string(privateKey)


    pk = SigningKey(b'\x9f\x1f\r\xab\xc6\x8bG [\xa6\x96\xf5\xeeJ\xc0"\xa3\x89\x18\xb4\xa2\xe0\xd1O\xa9\xce$\xe3\x98\xa9/\xf8')

    vk = VerifyKey(b'\xe9\x919rce\xc9\x1a\xcfJ}\xa3\xee\x17q\x19\xbd\x0eu\xf4\xe0\xd5\x8a<\xc0\x81\x0c\xdbD\xf5;G')

    #vk = b'6\x9a\x00\x8d\xf5\xa1$\x86\x8e\xabp\xb5d6\xb7\x1cY\xb3\xf9\xfc7ji\xb0\xfe@\xab\x85\x7fI8CtI(\xcdb\x99y%\x05\x1d\x02H\xae\x9b\xd2\xdd'
    #vk = VerifyingKey.from_string(vk)
    #vk.precompute()

    image_counter = ImageCounter()
    r = receiverlogic.Receiver(image_counter)

    # video info
    width = 416
    height = 416
    quality = 65

    # host computer info
    target_ip = '192.168.178.23'
    target_port = '5555'

    # statistics info
    moving_average_points = 50

    a = 0

    merkle_tree_interval = 127

    if merkle_tree_interval > 0:
        mt = MerkleTools()
        interval_count = 0

    # initialize sender
    image_sender = Sender(target_ip, target_port, pk)
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
    # image_count = 0
   # image_timer = []

    # responseTime = -1

    # streaming
    print('RPi Stream -> Start Streaming')
    while True:
        start_time = time.perf_counter()

        # time.sleep(0.005)

        # capture image
        image = rpi_cam.get_image()

        camera_time = time.perf_counter()

        # send compressed image (compress + send)
        compress_time, sign_time, send_time, = image_sender.send_image_compressed(
            image_counter.getInputCounter(), image)
        # image_timer.append(time.perf_counter())

        # queuesize = r.getSize()
      #  output = ''
     #   try:
     #       output, fetchDelay = r.get()
     #       responseTime = time.perf_counter() - (fetchDelay -
    #                                            output[1]) - image_timer[int(output[0][5:].split(':', 1)[0])]
     #   except:
     #       pass

        # boundingBoxes = boundingBoxes.decode().split(';')
        # if(len(boundingBoxes) != 2):
        #    for b in boundingBoxes:
        #        print(b)

        # postprocessing_time = time.perf_counter()
        # output =[]
        # verifying

        receive_time = time.perf_counter()

        responses = []
        # received = False
        # frames_behind = -1
        try:
            # output, fetchDelay = r.get()
            output = r.getAll()

            if merkle_tree_interval == 0:
                for o in output:
                        # received = True
                    sig = o.split(';--')[1].encode('latin1')
                    msg = o.split(';--')[0].encode('latin1')
                        # received_time = time.perf_counter() - receive_time

                    vk.verify(msg, sig)
                    responses.append(msg)
                #  receiverFrame = int(output[0][5:].split(':', 1)[0])
                #  frames_behind = image_count - receiverFrame
            else:

                for o in output:
                        # received = True
                    msg = o.split(';--')[0].encode('latin1')
                    try:
                        if image_counter.getOutputCounter() >= merkle_tree_interval:
                            root_hash = o.split(';--')[1].encode('latin1')
                            sig = o.split(';--')[2].encode('latin1')
                            match = vk.verify(root_hash, sig)
                            #print(match)
                            interval_count += 1


                    except:
                        pass

                    responses.append(msg)


        except:
            pass

        verify_time = time.perf_counter()

        for st in responses:
            if len(st) > 200:
                print(st)

        frames_behind = image_counter.getFramesAhead()
        # a = image_counter.getInputCounter(), image_counter.getOutputCounter(), image_counter.getFramesAhead()
        # print(a)

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
        # if received :
        # moving_average_receive_time.add(received_time)
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
