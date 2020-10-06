import Responder as re
from parameters import Parameters
from nacl.signing import SigningKey
import threading

class FrameSender:

    def __init__(self, hostname, sendingPort, merkle_tree_interval, contractHash):
        #self.hostname = hostname
        #self.port = port
        self._stop = False
        self._data = ''
        self._received = False
        self._readyToReceive = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(hostname, sendingPort, merkle_tree_interval, contractHash))        
        self._thread.daemon = True        
        self._thread.start()
        

    def putData(self, data, timeout =):
        flag = self._readyToReceive.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Contract aborted2222: Outsourcer at tcp://{}:{}".format(self.hostname, self.port) + 'timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review')

        #if waited :
            #print('Waited', (time.perf_counter() - a)*1000)

        self._data = data
        self._received = True
        self._readyToReceive.clear()


    def _run(merkle_tree_interval, contractHash):
        self._readyToReceive.set()
        sk = SigningKey(Parameters.private_key_contractor)

        responder = re.Responder(hostname, sendingPort)

        if merkle_tree_interval > 0:
            mt = MerkleTools()
            mtOld = MerkleTools()
            interval_count = 0
            mtOld_leaf_indices = {}
            mt_leaf_indices = {}
            #rendundancy_counter = 0
            #rendundancy_counter2 = 0
            current_challenge = 1
            merkle_root = ''
            #stringsend = ''
            last_challenge = 0


        while not self._stop:
            if self.received:
                self.received = False
                boxtext = self._data[0]
                image = self._data[1]
                if merkle_tree_interval == 0:
                    #sig = sk.sign_deterministic(boxtext.encode('latin1'))
                    sig = sk.sign(boxtext.encode('latin1') + contractHash).signature
                    #sig = list(sig)
                    sig = sig.decode('latin1')

                    # send reply

                    responder.respond(boxtext + ';--' + sig)

                else:
                    # print(image_count)
                    mt.add_leaf(boxtext, True) #add leafs dynamiclly to merkle tree
                    mt_leaf_indices[outsourcer_image_count] = image_count % merkle_tree_interval #remember indices for challenge
                    #print(image_count % merkle_tree_interval)
                    
                    
                    response = boxtext

                    # time to send a new merkle root
                    if image_count > 1 and (image_count+1) % merkle_tree_interval == 0: #e.g. if inervall = 128 then all respones from 0-127 are added to the merkle tree
                        #print(image_count)
                        a = time.perf_counter()
                        #rendundancy_counter = 2
                        mt.make_tree()
                        merkle_root = mt.get_merkle_root()

                        sig = sk.sign(merkle_root.encode(
                            'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                        # resond with merkle root
                        response += ';--' + str(merkle_root) + \
                            ';--' + sig.decode('latin1')

                        interval_count += 1
                        mtOld = mt  # save old merkle tree for challenge
                        #mtOld_leaf_indices.clear() # clear old indices
                        mtOld_leaf_indices.clear()
                        mtOld_leaf_indices = mt_leaf_indices.copy() #save old indices for challenge
                        #print(mtOld_leaf_indices)
                        mt_leaf_indices.clear() #clear for new indices
                        #mt_leaf_indices = {}

                        mt = MerkleTools()  # construct new merkle tree for next interval
                        te = time.perf_counter()-a
                    # print('1', te, image_count)
                    
                    else:
                        if interval_count > outsourcer_image_count : #if this is true then the outsourcer has not received the merkle root yet -> send again

                            sig = sk.sign(merkle_root.encode(
                            'latin1') + bytes(interval_count) + contractHash).signature  # sign merkle root

                            response += ';--' + str(merkle_root) + \
                            ';--' + sig.decode('latin1')

                        # print('2', image_count)

                        else: # in this case outsourcer has confirmed to have recieved the merkle root

                            if outsourcer_time_to_challenge and image_count - last_challenge > 3: #in this case outsourcer has sent a challenge to meet with the old merkle tree, give outsourcer 3 frames time to confirm challenge received before sending again
                                last_challenge = image_count
                                if outsourcer_random_number in mtOld_leaf_indices:
                                    outsourcer_random_number_index = mtOld_leaf_indices[outsourcer_random_number] #if challenge can be found, send proof back
                                
                                else:
                                    outsourcer_random_number_index = 0 #if challenge index cannot be found return leaf 0
                                    #print('proof index not found')


                                



                                

                                proofs = mtOld.get_proof(outsourcer_random_number_index)
                                
                                stringsend = ''
                                for proof in proofs:
                                    stringsend += ';--'  # indicate start of proof
                                    stringsend += proof.__str__()  # send proof

                                stringsend += ';--'
                                # send leaf
                                stringsend += mtOld.get_leaf(outsourcer_random_number_index)
                                stringsend += ';--'
                                stringsend += mtOld.get_merkle_root()  # send root

                                stringarr = []
                                stringarr = stringsend.split(';--')
                                
                                leaf_node = stringarr[-2]
                                root_node = stringarr[-1]
                                proof_string = stringarr[0:-2]

                                sig = sk.sign(str(stringarr[1:]).encode('latin1') + bytes(interval_count-1) + contractHash).signature  # sign proof and contract details
                                #print(str(stringarr).encode('latin1') + bytes(interval_count-1) + contractHash)
                                #print(stringarr)
                                    # attach signature
                                response += ';--' + sig.decode('latin1')
                                response += stringsend  # attach challenge response to response

                                


                                

                            # print('3', te, image_count)


                    responder.respond(response)

                response_signing_time = time.perf_counter()

            # print(response_signing_time- image_postprocessing_time)

                replied_time = time.perf_counter()

                # display image

                if not dont_show:
                    # image.show()

                    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                    cv2.imshow('raspberrypi', image)

                    if cv2.waitKey(1) == ord('q'):
                        responder.respond('abort12345:6')                
                        sys.exit(
                            'Contract aborted: Contractor ended contract according to custom')
                
                self._readyToReceive.set()