import time

from sender.rpi_camera import RPiCamera
from sender.sender_engine import Sender
from utilities.stats import MovingAverage


from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# from ecdsa import SigningKey

# from ecdsa import VerifyingKey

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

import random


def main():
    """
    main function interface
    :return: nothing
    """

    pk = SigningKey(Parameters.private_key_outsourcer)

    # print(pk.verify_key)

    vk = VerifyKey(OutsourceContract.public_key_contractor)

    # video info
    width = Parameters.input_size
    height = Parameters.input_size
    quality = Parameters.quality

    receiver_ip = Parameters.receiver_ip
    receiver_port = Parameters.receiver_port

    receiver_port_verifier = Parameters.receiver_port_verfier

    sending_port_verifier = Parameters.sending_port_verifier

    verifier_ip = Parameters.target_ip_verifier

    sending_port = Parameters.sending_port

    # statistics info
    moving_average_points = Parameters.moving_average_points

    

    merkle_tree_interval = OutsourceContract.merkle_tree_interval

    maxmium_number_of_frames_ahead = Parameters.maxmium_number_of_frames_ahead
    minimum_response_rate = Parameters.minimum_response_rate
    warm_up_time = Parameters.warm_up_time

    sampling_interval = Parameters.sampling_interval
    maxmium_number_of_frames_ahead_verifier = Parameters.maxmium_number_of_frames_ahead_verifier

    image_counter = ImageCounter(maxmium_number_of_frames_ahead)
    image_counter_verifier = ImageCounter(maxmium_number_of_frames_ahead)

    r = receiverlogic.Receiver(image_counter, receiver_ip, receiver_port)

    r_verifier = receiverlogic.Receiver(
        image_counter_verifier, receiver_ip, receiver_port_verifier)

    print('Waiting for contractor and verifier to connect ...')
    start_listening_time = time.perf_counter()
    while r.getConnectionEstablished() == False or r_verifier.getConnectionEstablished() == False:
        if time.perf_counter() - start_listening_time > 20:
            sys.exit(
                'Contract aborted: Contractor did not connect in time. Possible Consquences for Contractor: Blacklist, Bad Review')
        time.sleep(0.5)

    print('Connection with contractor and verfier established')

    a = 0

    outsourcerSample = (-1, '-1', '') #saves input_counter, response, signautre of current sample
    verifierSample = (-1, '-1', '') # saves input_counter, response, signature of current sample

    lastSample = -1 #last sample index that was compared

    contractHash = Helperfunctions.hashContract().encode('latin1')

    verifier_contract_hash = Helperfunctions.hashVerifierContract().encode('latin1')

    # sampling_start_count = 0 #saves image_count at the time of starting a new sample
    sampling_index = -1

    # interval_samples = sampling_interval * minimum_response_rate #interval to saves image samples

    if merkle_tree_interval > 0:
        mt = MerkleTools()
        interval_count = 0
        time_to_challenge = False
        random_number = random.randint(0, merkle_tree_interval - 1)
        current_root_hash = ''
    else:
        random_number = random.randint(0, sampling_interval - 1)

    # initialize sender
    image_sender = Sender(sending_port, pk, quality)
    image_sender.set_quality(quality)
    print('RPi Stream -> Sender Initialized')

    image_sender_verifier = Sender(sending_port_verifier, pk, quality)
    image_sender.set_quality(quality)
    print('RPi Stream -> Verifier Sender Initialized')

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

    moving_average_last_Sample = MovingAverage(moving_average_points)

    # streaming
    print('RPi Stream -> Start Streaming')
    while True:

        start_time = time.perf_counter()

        # capture image
        image = rpi_cam.get_image()

        camera_time = time.perf_counter()

        # if index is at random sample, send random sample to verifier
        if sampling_index == image_counter.getInputCounter():
            compress_time2, sign_time2, send_time2, = image_sender_verifier.send_image_compressed(
                image_counter.getInputCounter(), image, verifier_contract_hash, image_counter_verifier.getNumberofOutputsReceived())

        if merkle_tree_interval == 0:
            compress_time, sign_time, send_time, = image_sender.send_image_compressed(
                image_counter.getInputCounter(), image, contractHash, image_counter.getNumberofOutputsReceived())
        else:

            compress_time, sign_time, send_time = image_sender.send_image_compressed_Merkle(
                image_counter.getInputCounter(), image, contractHash, image_counter.getNumberofOutputsReceived(), random_number, interval_count, time_to_challenge)

        # verifying

        receive_time = time.perf_counter()

        responses = []
        signatures_outsourcer = [] 

        output = r.getAll()

        if merkle_tree_interval == 0:

            for o in output:

                try:
                    sig = o.split(';--')[1].encode('latin1')
                    msg = o.split(';--')[0].encode('latin1')
                except:
                    sys.exit(
                        'Contract aborted: Contractor response is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review')

                try:
                    vk.verify(msg + contractHash, sig)
                except:
                    sys.exit(
                        'Contract aborted: Contractor singature does not match response. Possible Consquences for Contractor: Blacklist, Bad Review')
                responses.append(msg)
                signatures_outsourcer.append(sig)

        else:  # Merkle tree verification is active

            if image_counter.getNumberofOutputsReceived() > (merkle_tree_interval) * (interval_count+2):
                sys.exit('Contract aborted: No root hash received for current interval in time. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

            for o in output:
                root_hash_received = False
                msg = o.split(';--')[0].encode('latin1')  # get output

                # section to check for proofs
                if time_to_challenge == True:  # If it's true then it's time to receive a challenge response
                    proof_received = False  # check if message structure indicates that it contains a proof
                    if len(o.split(';--')) > 3:
                        challenge_response = []
                        try:
                            signature = o.split(';--')[1].encode('latin1')
                            challenge_response = o.split(';--')[2:]
                            leaf_node = challenge_response[-2]
                            root_node = challenge_response[-1]
                            proof_string = challenge_response[0:-2]
                            proofList = []
                            for strings in proof_string:
                                strings = strings.replace("'", "\"")
                                proofList.append(json.loads(strings))
                            proof_received = True
                        except:
                            pass
                        if proof_received:  # if message contains a proof
                            # check if root node sent earlier matches current one
                            if current_root_hash == root_node.encode('latin1'):
                                mt = MerkleTools()

                                try:
                                    # print(str(challenge_response).encode('latin1') + bytes(interval_count-1) + contractHash)
                                    # print(challenge_response)
                                    vk.verify(str(challenge_response).encode(
                                        'latin1') + bytes(interval_count-1) + contractHash, signature)
                                except:
                                    sys.exit(
                                        'Contract aborted: Contractor singature of challenge response is incorrect. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

                                try:
                                    merkle_proof_of_membership = mt.validate_proof(
                                        proofList, leaf_node, root_node)  # verify proof of memebrship
                                    # print('Proof of membership for random sample in interval' + str(interval_count -1) + ' was successful')
                                except:
                                    merkle_proof_of_membership = False

                                if merkle_proof_of_membership == True:
                                    time_to_challenge = False  # all challenges passed
                                else:
                                    sys.exit(
                                        'Contract aborted: Leaf is not contained in Merkle Tree. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval, fine')

                            else:
                                sys.exit('Contract aborted: Contractor signature of root hash received at challenge response does not match previous signed root hash . Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval, fine')

                # section to check for merkle roots
                # if it's true then it's time to receive a new Merkle root
                if image_counter.getNumberofOutputsReceived() > (merkle_tree_interval) * (interval_count+1):

                    if time_to_challenge == True:
                        sys.exit('Contract aborted: Merkle Tree proof of membership challenge response was not received in time. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

                    try:  # check if merkle root received

                        root_hash = o.split(';--')[1].encode('latin1')
                        sig = o.split(';--')[2].encode('latin1')
                        if len(o.split(';--')) == 3:
                            root_hash_received = True
                    except:
                        pass

                    if root_hash_received == True:  # if root has received, verify signature

                        root_hash_received = False
                        time_to_challenge = True
                        random_number = random.randint(
                            0, merkle_tree_interval - 1)
                        try:

                            match = vk.verify(
                                root_hash + bytes(interval_count) + contractHash, sig)
                            interval_count += 1
                            current_root_hash = root_hash
                            # print(interval_count, image_counter.getNumberofOutputsReceived())
                        except:
                            sys.exit(
                                'Contract aborted: Contractor singature of root hash is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

                responses.append(msg)
                signatures_outsourcer.append(sig)

        
        
        
        responses_verifier = []
        signatures_verifier = []  
        output_verifier = r_verifier.getAll()

        for o in output_verifier:

            try:
                sig = o.split(';--')[1].encode('latin1')
                msg = o.split(';--')[0].encode('latin1')
            except:
                sys.exit(
                    'Contract aborted: Contractor response is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review')

            try:
                vk.verify(msg + verifier_contract_hash, sig)
            except:
                sys.exit(
                    'Contract aborted: Contractor singature does not match response. Possible Consquences for Contractor: Blacklist, Bad Review')
            responses_verifier.append(msg)
            signatures_verifier.append(sig)

        
        
        
        
        
        if image_counter.getNumberofOutputsReceived() % sampling_interval == 0:  # pick new random sample

            # random_number = random.randint(0,sampling_interval -1 - maxmium_number_of_frames_ahead)
            random_number = random.randint(0, sampling_interval - 1)
            sampling_index = random_number + image_counter.getInputCounter()

        if image_counter.getOutputCounter() == sampling_index and len(responses) > 0:
            outsourcerSample = (sampling_index, responses[-1], signatures_outsourcer[-1]  )
            

        if image_counter_verifier.getOutputCounter() == sampling_index and len(responses_verifier) > 0:
            verifierSample = (sampling_index, responses_verifier[-1], signatures_verifier[-1]  )


        if outsourcerSample[0] == verifierSample[0]:
            #compare resp  
            if lastSample != outsourcerSample[0]:
                lastSample = outsourcerSample[0]
                if outsourcerSample[1] == verifierSample[1]:
                    print(True, outsourcerSample[1])
                else:
                    print(False, outsourcerSample[1], verifierSample[1])  

        

        verify_time = time.perf_counter()
        if(OutsourceContract.criteria == 'Atleast 2 objects detected'):
            for st in responses:
                if len(st) > 1000:
                    print(st)

        frames_behind = image_counter.getFramesAhead()

        if frames_behind > maxmium_number_of_frames_ahead:
            if image_counter.getInputCounter() > warm_up_time:
                # print(image_counter.getInputCounter(), image_counter.getFramesAhead())

                sys.exit(
                    'Contract aborted: Contractor response delay rate is too high. Possible Consquences for Contractor: Bad Review, Blacklist')

        if(image_counter.getNumberofOutputsReceived() < image_counter.getInputCounter() * minimum_response_rate):
            if image_counter.getInputCounter() > warm_up_time:
                sys.exit(
                    'Contract aborted: Contractor response rate is too low. Possible Consquences for Contractor: Bad Review, Blacklist')

        # if(image_counter.getInputCounter() - image_counter.getOutputCounter() < 60):
        #     if image_counter.getInputCounter() > 1000:
        #         print('b', image_counter.getOutputCounter() - image_counter.getInputCounter())

        # if frames_behind > 60:
        #     print('b', frames_behind)

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



        moving_average_last_Sample.add(image_counter.getInputCounter() - lastSample)
        

        total_time = moving_average_camera_time.get_moving_average() \
            + moving_average_compress_time.get_moving_average() \
            + moving_average_sign_time.get_moving_average() \
            + moving_average_send_time.get_moving_average()

        instant_fps = 1 / (time.perf_counter() - start_time)
        moving_average_fps.add(instant_fps)

        # terminal prints
        if image_counter.getInputCounter() % 20 == 0:
            print("total: %5.1fms (%5.1ffps) camera %4.1f (%4.1f%%) compressing %4.1f (%4.1f%%) signing %4.1f (%4.1f%%) sending %4.1f (%4.1f%%) frames ahead %4.1f last sample %4.1f verify time %4.1f (%4.1f%%) "
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

                      moving_average_last_Sample.get_moving_average(),

                      moving_average_verify_time.get_moving_average()*1000,
                      moving_average_verify_time.get_moving_average() / total_time * 100), end='\r')

        # counter
        image_counter.increaseInputCounter()

       # print(c - a)
        # a = time.perf_counter()


if __name__ == '__main__':
    main()
