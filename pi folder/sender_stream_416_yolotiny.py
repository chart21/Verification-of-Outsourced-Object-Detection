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


    maxmium_number_of_verifier_sample_missed_consecutively = Parameters.maxmium_number_of_verifier_sample_missed_consecutively
    minimum_response_rate_verifier = Parameters.minimum_response_rate_verifier

    framesync = Parameters.framesync

    image_counter = ImageCounter(maxmium_number_of_frames_ahead)
    image_counter_verifier = ImageCounter(maxmium_number_of_frames_ahead)

    r = receiverlogic.Receiver(image_counter, receiver_ip, receiver_port)

    r_verifier = receiverlogic.Receiver(
        image_counter_verifier, receiver_ip, receiver_port_verifier)

    print('Waiting for contractor and verifier to connect ...')
    start_listening_time = time.perf_counter()
    while r.getConnectionEstablished() == False or r_verifier.getConnectionEstablished() == False:
        if time.perf_counter() - start_listening_time > 35:
            r.close()
            r_verifier.close()
            time.sleep(1)
            sys.exit(
                'Contract aborted: Contractor did not connect in time. Possible Consquences for Contractor: Blacklist, Bad Review')
        time.sleep(0.5)

    print('Connection with contractor and verfier established')

    a = 0

    # saves input_counter, response, signautre of current sample
    outsourcerSample = (-1, '-1', '')
    # saves input_counter, response, signature of current sample
    verifierSample = (-1, '-1', '')

    outsourcer_sample_dict = {}
    verifier_sample_dict = {}

    output_counter = 0
    output_counter_verifier = 0

    lastSample = -1  # last sample index that was compared
    #lastVerifierSample = -1 #
    #lastOutsourcerSample = -1
    saved_compressed_sample_image = b'' #stores the compressed last sample image. If signed, unmatching responses are received any third party can verify with this saved image which response is not ocrrect

    contractHash = Helperfunctions.hashContract().encode('latin1')

    verifier_contract_hash = Helperfunctions.hashVerifierContract().encode('latin1')

    # sampling_start_count = 0 #saves image_count at the time of starting a new sample
    sampling_index = -1

    # interval_samples = sampling_interval * minimum_response_rate #interval to saves image samples

    verifier_sample_processed = 0
    verifier_sample_missed = 0 #how many samples were missed in total
    verifier_sample_missed_consecutively = 0 #how many samples were missed conecutively
    
    if merkle_tree_interval > 0:
        mt = MerkleTools()
        interval_count = 0
        time_to_challenge = False
        next_merkle_chall = 0
        curr_merkle_chall = 0
        #random_number = random.randint(0, merkle_tree_interval - 1)
        current_root_hash = ''
        next_merkle_response = ''
        curr_merkle_response = ''
        sample_received_in_interval = -2
        abort_at_next_merkle_root = False

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

        if framesync:
            if image_counter.getInputCounter() > warm_up_time:
                milisecs_outsourcer = moving_average_fps.get_moving_average()
                frames_ahead_average = moving_average_response_time.get_moving_average()
                adjusted_milisecs = (frames_ahead_average-1)* (1/milisecs_outsourcer) - 0.005 #add safety puffer
                #adjusted_milisecs -= 0.01
                if adjusted_milisecs > 0:
                    time.sleep(adjusted_milisecs)


        camera_time = time.perf_counter()

        # if index is at random sample, send random sample to verifier

        if merkle_tree_interval == 0:


            if sampling_index == image_counter.getInputCounter(): #send image to both outsourcer and verifier
                compress_time, sign_time, send_time, compressed = image_sender.send_image_compressed_with_return(
                    image_counter.getInputCounter(), image, contractHash, image_counter.getNumberofOutputsReceived())  
                
                compress_time2, sign_time2, send_time2, = image_sender_verifier.send_image_compressed_with_input(
                    image_counter.getInputCounter(), image, verifier_contract_hash, image_counter_verifier.getNumberofOutputsReceived(), compressed)
                
                saved_compressed_sample_image = compressed
                
                compress_time += compress_time2
                sign_time += sign_time2
                send_time +=send_time2

            else:
                compress_time, sign_time, send_time = image_sender.send_image_compressed(
                    image_counter.getInputCounter(), image, contractHash, image_counter.getNumberofOutputsReceived())  


        else:    
            
            
            # reuse compressed image and just add signature
            # if sampling_index == image_counter.getInputCounter() or sampling_index + sampling_interval < image_counter.getInputCounter(): # only for high frequency to reduce receive time
            if sampling_index == image_counter.getInputCounter(): #send image to both outsourcer and verifier
                
                compress_time, sign_time, send_time, compressed = image_sender.send_image_compressed_Merkle_with_return(
                    image_counter.getInputCounter(), image, contractHash, image_counter.getNumberofOutputsReceived(), curr_merkle_chall, interval_count, time_to_challenge)


                
                compress_time2, sign_time2, send_time2 = image_sender_verifier.send_image_compressed_with_input(
                    image_counter.getInputCounter(), image, verifier_contract_hash, image_counter_verifier.getNumberofOutputsReceived(), compressed)

                saved_compressed_sample_image = compressed

                compress_time += compress_time2
                sign_time += sign_time2
                send_time += send_time2
            else:

                compress_time, sign_time, send_time = image_sender.send_image_compressed_Merkle(
                    image_counter.getInputCounter(), image, contractHash, image_counter.getNumberofOutputsReceived(), curr_merkle_chall, interval_count, time_to_challenge)



        # verifying

        receive_time = time.perf_counter()

        responses = []
        signatures_outsourcer = []

        output = r.getAll()

        if merkle_tree_interval == 0:

            for o in output:
                if o[:5] == 'abort':
                    image_sender_verifier.send_abort(image)
                    r.close()
                    r_verifier.close()
                    time.sleep(1)
                    sys.exit('Contract aborted by contractor according to custom') 

                try:
                    sig = o.split(';--')[1].encode('latin1')
                    msg = o.split(';--')[0].encode('latin1')
                except:
                    r.close()
                    r_verifier.close()
                    time.sleep(1)
                    sys.exit(
                        'Contract aborted: Contractor response is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review')

                try:
                    vk.verify(msg + contractHash, sig)
                except:
                    r.close()
                    r_verifier.close()
                    time.sleep(1)
                    sys.exit(
                        'Contract aborted: Contractor singature does not match response. Possible Consquences for Contractor: Blacklist, Bad Review')
                responses.append(msg)
                signatures_outsourcer.append(sig)

        else:  # Merkle tree verification is active

            if image_counter.getNumberofOutputsReceived() > (merkle_tree_interval) * (interval_count+2):
                r.close()
                r_verifier.close()
                time.sleep(1)
                sys.exit('Contract aborted: No root hash received for current interval in time. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

            for o in output:
                if o[:5] == 'abort':                
                    image_sender_verifier.send_abort(image)
                    r.close()
                    r_verifier.close()
                    time.sleep(1)
                    sys.exit('Contract aborted by contractor according to custom') 

                root_hash_received = False
                msg = o.split(';--')[0].encode('latin1')  # get output

                # section to check for proofs
                if time_to_challenge == True:  # If it's true then it's time to receive a challenge response
                    proof_received = False  # check if message structure indicates that it contains a proof
                    if len(o.split(';--')) > 3:
                        challenge_response = []
                        try:
                            
                            signature = o.split(';--')[1].encode('latin1')
                            signatures_outsourcer.append(signature)
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
                            mt = MerkleTools()
                            
                            if sample_received_in_interval == interval_count -1: #skip this part of the challenge if no sample was compared in last interval count
                                mt.add_leaf(curr_merkle_response.decode('latin1'), True) #get last response into same format as leaf_node
                                if leaf_node != mt.get_leaf(0):
                                    print('Merkle tree leaf node does not match earlier sent response')
                                #else:
                                    #print('Success')
                            
                            
                            if current_root_hash == root_node.encode('latin1'): #check if signed root hash received earlier equals sent root hash
                                

                                try:
                                    # print(str(challenge_response).encode('latin1') + bytes(interval_count-1) + contractHash)
                                    # print(challenge_response)
                                    vk.verify(str(challenge_response).encode(
                                        'latin1') + bytes(interval_count-1) + contractHash, signature)
                                except:
                                    r.close()
                                    r_verifier.close()
                                    time.sleep(1)
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
                                    r.close()
                                    r_verifier.close()
                                    time.sleep(1)
                                    sys.exit(
                                        'Contract aborted: Leaf is not contained in Merkle Tree. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval, fine')

                            else:
                                r.close()
                                r_verifier.close()
                                time.sleep(1)
                                sys.exit('Contract aborted: Contractor signature of root hash received at challenge response does not match previous signed root hash . Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval, fine')

                # section to check for merkle roots
                # if it's true then it's time to receive a new Merkle root
                if image_counter.getNumberofOutputsReceived() >= (merkle_tree_interval) * (interval_count+1):

                    if time_to_challenge == True:
                        r.close()
                        r_verifier.close()
                        time.sleep(1)
                        sys.exit('Contract aborted: Merkle Tree proof of membership challenge response was not received in time. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

                    try:  # check if merkle root received

                        root_hash = o.split(';--')[1].encode('latin1')
                        sig = o.split(';--')[2].encode('latin1')
                        if len(o.split(';--')) == 3:
                            root_hash_received = True
                    except:
                        pass

                    if root_hash_received == True:  # if root hash received, verify signature

                        root_hash_received = False
                        time_to_challenge = True
                        random_number = random.randint(
                            0, merkle_tree_interval - 1)
                        try:

                            match = vk.verify(
                                root_hash + bytes(interval_count) + contractHash, sig)
                            interval_count += 1
                            curr_merkle_chall = next_merkle_chall #to send last checked sample as challenge 
                            current_root_hash = root_hash
                            curr_merkle_response = next_merkle_response  #to remmeber last check response
                            # print(interval_count, image_counter.getNumberofOutputsReceived())
                        except:
                            r.close()
                            r_verifier.close()
                            time.sleep(1)
                            sys.exit(
                                'Contract aborted: Contractor singature of root hash is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review, Refuse of Payment for images from current interval')

                    
                        if abort_at_next_merkle_root:
                            r.close()
                            r_verifier.close()
                            time.sleep(1)
                            sys.exit(
                                'Contract aborted: Merkle Tree is built on responses unequal to responses of the verifier. Possible Consquences for Contractor: Fine, Blacklist, Bad Review')

                    
                    
                    
                    
                    
                    signatures_outsourcer.append(sig)
                responses.append(msg)
                if len(signatures_outsourcer) == 0:
                    signatures_outsourcer.append('Next merkle root serves as a proof')
                

        responses_verifier = []
        signatures_verifier = []
        output_verifier = r_verifier.getAll()

        for o in output_verifier:
            if o[:5] == 'abort':
                image_sender.send_abort(image)
                r.close()
                r_verifier.close()
                time.sleep(1)
                sys.exit('Contract aborted by verfier according to custom') 
            try:
                sig = o.split(';--')[1].encode('latin1')
                msg = o.split(';--')[0].encode('latin1')
            except:
                r.close()
                r_verifier.close()
                time.sleep(1)
                sys.exit(
                    'Contract aborted: Contractor response is ill formated. Possible Consquences for Contractor: Blacklist, Bad Review')

            try:
                vk.verify(msg + verifier_contract_hash, sig)
            except:
                r.close()
                r_verifier.close()
                time.sleep(1)
                sys.exit(                    
                    'Contract aborted: Contractor singature does not match response. Possible Consquences for Contractor: Blacklist, Bad Review')
            responses_verifier.append(msg)
            signatures_verifier.append(sig)

        # make sure outspurcer has even computed a new output before assigning a new sample, otherwise it's possible to never compare samples
        if image_counter.getOutputCounter() == sampling_index and len(responses) > 0:

            if int(responses[-1].decode('latin1')[5:].split(':', 1)[0]) == sampling_index: #in rare cases of threading timing output counter and responses can be desynced
                outsourcer_sample_dict[sampling_index] = (
                    sampling_index, responses[-1], signatures_outsourcer[-1])
                #merkle_challenge_index = image_counter.getOutputCounter() % merkle_tree_interval

        if image_counter_verifier.getOutputCounter() == sampling_index and len(responses_verifier) > 0:
            # make sure verfier has even computed a new output beofre assigning a new sample, otherwise it's possible to never compare samples
            if int(responses_verifier[-1].decode('latin1')[5:].split(':', 1)[0]) == sampling_index: #in rare cases of threading timing output counter and responses can be desynced
                verifier_sample_dict[sampling_index] = (
                    sampling_index, responses_verifier[-1], signatures_verifier[-1])
            

        # sample_checked = False
        # if outsourcerSample[0] == verifierSample[0]:
        #     sample_checked = True
        #     #compare resp
        #     if lastSample != outsourcerSample[0]:
        #         lastSample = outsourcerSample[0]
        #         if outsourcerSample[1] == verifierSample[1]:
        #             print(True, outsourcerSample[1])
        #         else:
        #             print(False, outsourcerSample[1], verifierSample[1])

        sample_checked = False
        if sampling_index in verifier_sample_dict and sampling_index in outsourcer_sample_dict:
            if outsourcer_sample_dict[sampling_index][0] == verifier_sample_dict[sampling_index][0]:
                sample_checked = True
                
                # compare resp
                if lastSample != outsourcer_sample_dict[sampling_index][0]:
                    lastSample = outsourcer_sample_dict[sampling_index][0]
                    if outsourcer_sample_dict[sampling_index][1] == verifier_sample_dict[sampling_index][1]:
                        #print(True, outsourcer_sample_dict[sampling_index][1])
 
                        if merkle_tree_interval > 0:
                            next_merkle_chall = outsourcer_sample_dict[sampling_index][0]
                            next_merkle_response = outsourcer_sample_dict[sampling_index][1]
                            
                            sample_received_in_interval = interval_count  #used to check if a sample was received in current merkle interval
                       
                        
                        outsourcer_sample_dict.clear()
                        verifier_sample_dict.clear()
                        #outsourcer_sample_dict = {} not needed since keys dont repeat
                        #verifier_sample_dict = {}  

                    else: #sample was found to be not equal
                        
                        if merkle_tree_interval == 0:
                            r.close()
                            r_verifier.close()
                            time.sleep(1)
                            sys.exit('Contract aborted. The following outputs are not equal: Outsourcer: ' +str(outsourcer_sample_dict[sampling_index][1]) +
                            ' , Verifier: ' + str(verifier_sample_dict[sampling_index][1]) + '  Possible consequences for cheating party: Fine, Blacklist, Bad Review '
                             )
                        else:
                            print(
                            "The following outputs are not equal:", outsourcer_sample_dict[sampling_index][1], verifier_sample_dict[sampling_index][1]) #if no merkle tree -> exit, if merkle tree wait for next chall
                            abort_at_next_merkle_root = True

        if image_counter.getNumberofOutputsReceived() % sampling_interval == 0:  # pick new random sample
            # only pick next sample if both parties have already processed last sample
            if image_counter_verifier.getOutputCounter() >= sampling_index and image_counter.getOutputCounter() >= sampling_index:
                
                # random_number = random.randint(0,sampling_interval -1 - maxmium_number_of_frames_ahead)
                random_number = random.randint(1, sampling_interval)
                sampling_index = random_number + image_counter.getInputCounter()
                verifier_sample_processed += 1 #save that verifier sucessfully processed sample
                verifier_sample_missed_consecutively = 0 #reset
            else:
                if image_counter.getInputCounter() - sampling_index > maxmium_number_of_frames_ahead_verifier: #means that verifier has lost sample or is too slow
                    random_number = random.randint(1, sampling_interval)
                    sampling_index = random_number + image_counter.getInputCounter()
                    verifier_sample_missed+=1 #save that verifier missed sample because of frame loss or being too slow
                    verifier_sample_missed_consecutively+=1
                    if image_counter.getInputCounter() > warm_up_time:
                        if verifier_sample_missed_consecutively > maxmium_number_of_verifier_sample_missed_consecutively or verifier_sample_missed/verifier_sample_processed < minimum_response_rate_verifier :
                            r.close()
                            r_verifier.close()
                            time.sleep(1)
                            sys.exit(
                    'Contract aborted: Verifier has failed to process enough samples in time. Possible Consquences for Verifier: Bad Review, Blacklist')

                

        
        
        #image_sender.manualCancel()
        
        
        verify_time = time.perf_counter()
        if(OutsourceContract.criteria == 'Atleast 2 objects detected'):
            for st in responses:
                if len(st) > 1000:
                    print(st)

        frames_behind = image_counter.getFramesAhead()

        if frames_behind > maxmium_number_of_frames_ahead:
            if image_counter.getInputCounter() > warm_up_time:
                # print(image_counter.getInputCounter(), image_counter.getFramesAhead())
                r.close()
                r_verifier.close()
                time.sleep(1)
                sys.exit(
                    'Contract aborted: Contractor response delay rate is too high. Possible Consquences for Contractor: Bad Review, Blacklist')

        if(image_counter.getNumberofOutputsReceived() < image_counter.getInputCounter() * minimum_response_rate):
            if image_counter.getInputCounter() > warm_up_time:
                r.close()
                r_verifier.close()
                time.sleep(1)
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
            print('contractor', a)

        if(image_counter_verifier.getNumberofOutputsReceived() == 300):
            b = time.perf_counter()
        if(image_counter_verifier.getNumberofOutputsReceived() == 700):
            b = time.perf_counter() - b
            print('verifier', b)

        # statistics
        moving_average_camera_time.add(camera_time - start_time)
        moving_average_compress_time.add(compress_time)
        moving_average_sign_time.add(sign_time)
        moving_average_send_time.add(send_time)

        moving_average_verify_time.add(verify_time - receive_time)
        if(frames_behind != -1):
            moving_average_response_time.add(frames_behind)

        if sample_checked:
            moving_average_last_Sample.add(
                image_counter.getInputCounter() - lastSample)

        total_time = moving_average_camera_time.get_moving_average() \
            + moving_average_compress_time.get_moving_average() \
            + moving_average_sign_time.get_moving_average() \
            + moving_average_send_time.get_moving_average()

        instant_fps = 1 / (time.perf_counter() - start_time)
        moving_average_fps.add(instant_fps)

        # terminal prints
        if image_counter.getInputCounter() % 20 == 0:
            print("total: %5.1fms (%5.1ffps) camera %4.1f (%4.1f%%) compressing %4.1f (%4.1f%%) signing %4.1f (%4.1f%%) sending %4.1f (%4.1f%%) frames ahead %4.1f ahead of sample %4.1f verify time %4.1f (%4.1f%%) "
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
