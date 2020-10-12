# Main class of a contractor or verifier using an Edge TPU without the use of threading
# Paramters associated with this class including if this device should act as a contractor or verifier can be set in parameters.py
from parameters import ParticipantData
from parameters import Parameters
from parameters import OutsourceContract
from parameters import VerifierContract
from parameters import Helperfunctions
import json
from merkletools import MerkleTools
import sys
import videoStramSubscriber as vss
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import time
import imagezmq
import Responder as re

import numpy as np
import cv2
from PIL import Image

from absl import app, flags, logging

import os
# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utilities.stats import MovingAverage
from object_detection.object_detection import Model
from utilities.render import Render


def main(_argv):

    # get paramters and contract details

    if Parameters.is_contractor == True:  # checks if this machine is outsourcer or verifier
        vk = VerifyKey(OutsourceContract.public_key_outsourcer)
        contractHash = Helperfunctions.hashContract().encode('latin1')
        model_to_use = OutsourceContract.model
        tiny = OutsourceContract.tiny
        merkle_tree_interval = OutsourceContract.merkle_tree_interval
    else:
        vk = VerifyKey(VerifierContract.public_key_outsourcer)
        contractHash = Helperfunctions.hashVerifierContract().encode('latin1')
        model_to_use = VerifierContract.model
        tiny = VerifierContract.tiny
        merkle_tree_interval = 0

    sk = SigningKey(Parameters.private_key_self)
    framework = Parameters.framework
    weights = Parameters.weights
    count = Parameters.count
    dont_show = Parameters.dont_show
    info = Parameters.info
    crop = Parameters.crop
    input_size = Parameters.input_size
    iou = Parameters.iou
    score = Parameters.score

    hostname = Parameters.ip_outsourcer  # Use to receive from other computer
    port = Parameters.port_outsourcer
    sendingPort = Parameters.sendingPort
    minimum_receive_rate_from_contractor = Parameters.minimum_receive_rate_from_contractor

    edgeTPU_model_path = Parameters.edgeTPU_model_path
    edgeTPU_label_path = Parameters.edgeTPU_label_Path
    edgeTPU_confidence_level = Parameters.EdgeTPU_confidence_level

    # configure video stream receiver

    receiver = vss.VideoStreamSubscriber(hostname, port)
    print('Receiver Initialized')

    # configure model

    model = Model()
    model.load_model(edgeTPU_model_path)
    model.load_labels(edgeTPU_label_path)
    model.set_confidence_level(edgeTPU_confidence_level)

    # configure responder
    responder = re.Responder(hostname, sendingPort)

    render = Render()

    # configure and iniitialize statistic variables

    moving_average_points = 50

    moving_average_fps = MovingAverage(moving_average_points)
    moving_average_receive_time = MovingAverage(moving_average_points)
    moving_average_decompress_time = MovingAverage(moving_average_points)

    moving_average_img_preprocessing_time = MovingAverage(
        moving_average_points)

    moving_average_model_inference_time = MovingAverage(moving_average_points)

    moving_average_img_postprocessing_time = MovingAverage(
        moving_average_points)

    moving_average_reply_time = MovingAverage(moving_average_points)
    moving_average_image_show_time = MovingAverage(moving_average_points)
    moving_average_verify_image_sig_time = MovingAverage(moving_average_points)

    moving_average_response_signing_time = MovingAverage(moving_average_points)

    image_count = 0

    a = 0
    b = 0

    # configure Merkle tree related variables if merkle trees are to be used

    if merkle_tree_interval > 0:
        mt = MerkleTools()
        mtOld = MerkleTools()
        interval_count = 0
        mtOld_leaf_indices = {}
        mt_leaf_indices = {}
        current_challenge = 1
        merkle_root = ''
        last_challenge = 0

    while True:

        start_time = time.perf_counter()

        # receive image

        name, compressed = receiver.receive()

        if name == 'abort':
            sys.exit('Contract aborted by outsourcer according to custom')

        received_time = time.perf_counter()

        # decompress image

        decompressedImage = cv2.imdecode(
            np.frombuffer(compressed, dtype='uint8'), -1)

        decompressed_time = time.perf_counter()

        # verify image  (verify if signature matches image, contract hash and image count, and number of outptuts received)
        if merkle_tree_interval == 0:
            try:
                vk.verify(bytes(compressed) + contractHash +
                          bytes(name[-2]) + bytes(name[-1]), bytes(name[:-2]))
            except:
                sys.exit(
                    'Contract aborted: Outsourcer signature does not match input. Possible Consquences for Outsourcer: Blacklist, Bad Review')

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

        verify_time = time.perf_counter()

        # image preprocessing

        model.load_image_cv2_backend(decompressedImage)

        image_preprocessing_time = time.perf_counter()

        # inference

        class_ids, scores, boxes = model.inference()

        model_inferenced_time = time.perf_counter()

        # image postprocessing

        render.set_image(decompressedImage)
        boxtext = render.render_detection(
            model.labels, class_ids, boxes, decompressedImage.shape[1], decompressedImage.shape[0], (45, 227, 227), 3)

        if merkle_tree_interval == 0:
            boxtext = 'Image' + str(name[-2]) + ':;' + boxtext
        else:
            boxtext = 'Image' + str(outsourcer_image_count) + ':;' + boxtext

        image_postprocessing_time = time.perf_counter()

        if merkle_tree_interval == 0:
            sig = sk.sign(boxtext.encode('latin1') + contractHash).signature
            sig = sig.decode('latin1')

            # send reply

            responder.respond(boxtext + ';--' + sig)

        else:
            mt.add_leaf(boxtext, True)  # add leafs dynamiclly to merkle tree
            # remember indices for challenge
            mt_leaf_indices[outsourcer_image_count] = image_count % merkle_tree_interval

            response = boxtext

            # if statement is true then it's time to send a new merkle root
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
                # mtOld_leaf_indices.clear() # clear old indices
                mtOld_leaf_indices.clear()
                mtOld_leaf_indices = mt_leaf_indices.copy()  # save old indices for challenge

                mt_leaf_indices.clear()  # clear for new indices

                mt = MerkleTools()  # construct new merkle tree for next interval

            else:
                # if statement is true then it's time to resend the merkle root because outsourcer has not received it yet
                # if this is true then the outsourcer has not received the merkle root yet -> send again
                if interval_count > outsourcer_image_count:

                    sig = sk.sign(merkle_root.encode(
                        'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                    response += ';--' + str(merkle_root) + \
                        ';--' + sig.decode('latin1')

                else:  # in this case outsourcer has confirmed to have recieved the merkle root

                    # if statement is true then it's time to resond to a challenge from the outsourcer
                    # in this case outsourcer has sent a challenge to meet with the old merkle tree, give outsourcer 3 frames time to confirm challenge received before sending again
                    if outsourcer_time_to_challenge and image_count - last_challenge > 3:
                        last_challenge = image_count
                        if outsourcer_random_number in mtOld_leaf_indices:
                            # if challenge can be found, send proof back
                            outsourcer_random_number_index = mtOld_leaf_indices[outsourcer_random_number]

                        else:
                            # if challenge index cannot be found return leaf 0
                            outsourcer_random_number_index = 0
                            #print('proof index not found')

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

        response_signing_time = time.perf_counter()

        replied_time = time.perf_counter()

        # display image

        if not dont_show:
            cv2.imshow('raspberrypi', decompressedImage)

            if cv2.waitKey(1) == ord('q'):
                responder.respond('abort12345:6')
                sys.exit(
                    'Contract aborted: Contractor ended contract according to custom')

        image_showed_time = time.perf_counter()

        # statistics

        moving_average_fps.add(1 / (image_showed_time - start_time))

        moving_average_receive_time.add(received_time - start_time)

        moving_average_decompress_time.add(decompressed_time - received_time)

        moving_average_verify_image_sig_time.add(
            verify_time - decompressed_time)

        moving_average_img_preprocessing_time.add(
            image_preprocessing_time - verify_time)

        moving_average_model_inference_time.add(
            model_inferenced_time - image_preprocessing_time)

        moving_average_img_postprocessing_time.add(
            image_postprocessing_time - model_inferenced_time)

        moving_average_response_signing_time.add(
            response_signing_time - image_postprocessing_time)  # adjust for merkle root

        moving_average_reply_time.add(replied_time - response_signing_time)

        moving_average_image_show_time.add(image_showed_time - replied_time)

        total_time = moving_average_receive_time.get_moving_average() \
            + moving_average_decompress_time.get_moving_average() \
            + moving_average_verify_image_sig_time.get_moving_average() \
            + moving_average_img_preprocessing_time.get_moving_average() \
            + moving_average_model_inference_time.get_moving_average() \
            + moving_average_img_postprocessing_time.get_moving_average() \
            + moving_average_response_signing_time.get_moving_average() \
            + moving_average_reply_time.get_moving_average() \
            + moving_average_image_show_time.get_moving_average()

        # count seconds it takes to process 400 images after a 800 frames warm-up time
        if(image_count == 800):
            a = time.perf_counter()
        if(image_count == 1200):
            a = time.perf_counter() - a
            print(a)

        # terminal prints
        if image_count % 20 == 0:

            print(" total: %4.1fms (%4.1ffps) "
                  " receiving %4.1f (%4.1f%%) "
                  " decoding %4.1f (%4.1f%%) "
                  " verifying %4.1f (%4.1f%%) "
                  " preprocessing %4.1f (%4.1f%%) "
                  " model inference %4.1f (%4.1f%%) "
                  " postprocessing %4.1f (%4.1f%%) "
                  " signing %4.1f (%4.1f%%) "
                  " replying %4.1f (%4.1f%%) "
                  " display %4.1f (%4.1f%%) "
                  % (
                      1000/moving_average_fps.get_moving_average(),
                      moving_average_fps.get_moving_average(),



                      moving_average_receive_time.get_moving_average()*1000,
                      moving_average_receive_time.get_moving_average() / total_time * 100,

                      moving_average_decompress_time.get_moving_average()*1000,
                      moving_average_decompress_time.get_moving_average() / total_time * 100,




                      moving_average_verify_image_sig_time.get_moving_average()*1000,
                      moving_average_verify_image_sig_time.get_moving_average() / total_time * 100,

                      moving_average_img_preprocessing_time.get_moving_average()*1000,
                      moving_average_img_preprocessing_time.get_moving_average() / total_time * 100,


                      moving_average_model_inference_time.get_moving_average()*1000,
                      moving_average_model_inference_time.get_moving_average() / total_time * 100,

                      moving_average_img_postprocessing_time.get_moving_average()*1000,
                      moving_average_img_postprocessing_time.get_moving_average() / total_time * 100,

                      moving_average_response_signing_time.get_moving_average()*1000,
                      moving_average_response_signing_time.get_moving_average() / total_time * 100,



                      moving_average_reply_time.get_moving_average() * 1000,
                      moving_average_reply_time.get_moving_average() / total_time * 100,


                      moving_average_image_show_time.get_moving_average()*1000,
                      moving_average_image_show_time.get_moving_average() / total_time * 100,), end='\r')

        # counter
        image_count += 1


if __name__ == '__main__':
    app.run(main)
