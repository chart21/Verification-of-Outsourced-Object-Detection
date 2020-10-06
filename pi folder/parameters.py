import hashlib

class OutsourceContract:
    contract_uid = 0 #contracts have to have a unique id to esnure that each contract hash is unique
    public_key_outsourcer = b'e\x0fy\xfd\xe6\x16\x1f\xe0\x16B\xf2\xdb\x1d\x7f\xc9\xbcLCo\xa7\xa6c\x17\xbf\x8fo\xc8[\x07|bL'
    public_key_contractor = b'\xe9\x919rce\xc9\x1a\xcfJ}\xa3\xee\x17q\x19\xbd\x0eu\xf4\xe0\xd5\x8a<\xc0\x81\x0c\xdbD\xf5;G'
    reward_per_image = 0
    deposit_outsourcer = 0  #deposit of outsourcer to ensure paying fine and reward is possible
    deposit_contractor = 0 #deposit of contractor to ensure paying fine
    fine_outsourcer = 0  #fine if a party is detected cheating
    fine_contractor = 0
    model = 'yolov4' #model to use, possible choices are yolov4, yolov3
    tiny = True #whether to use tiny weigths for higher performance
    merkle_tree_interval = 0 # 0: Do not use Merkle Tree but sing every output image, >0: Specifies the intervals at wich a Merkle Tree root is signed and sent
    criteria = 'Atleast 2 objects detected'   #Specifies if all outputs should be sent back or only outputs that fulfill a certain criteria (e.g certain event happens), criterias should be combined with Merkle Trees to ensure overall consistency
    deposit_verfier = 10000000
    fine_verifier = 500000
    reward_per_image_verifier = 1
    
class VerifierContract:
    contract_uid = 0 #contracts have to have a unique id to esnure that each contract hash is unique
    public_key_outsourcer = b'e\x0fy\xfd\xe6\x16\x1f\xe0\x16B\xf2\xdb\x1d\x7f\xc9\xbcLCo\xa7\xa6c\x17\xbf\x8fo\xc8[\x07|bL'
    public_key_verifier = b'\xe9\x919rce\xc9\x1a\xcfJ}\xa3\xee\x17q\x19\xbd\x0eu\xf4\xe0\xd5\x8a<\xc0\x81\x0c\xdbD\xf5;G'

    deposit_verfier = 10000000
    fine_verifier = 500000
    reward_per_image_verifier = 1
    
    model = 'yolov4' #model to use, possible choices are yolov4, yolov3
    tiny = True #whether to use tiny weigths for higher performance

    deposit_verfier = 10000000
    fine_verifier = 500000
    reward_per_image_verifier = 1



class Parameters:
    receiver_ip = "192.168.178.34" #outsourcer, own ip to initialize image hub
    sending_port = 5555 #port to send images to
    sending_port_verifier = 5556
    private_key_outsourcer = b'\x9f\x1f\r\xab\xc6\x8bG [\xa6\x96\xf5\xeeJ\xc0"\xa3\x89\x18\xb4\xa2\xe0\xd1O\xa9\xce$\xe3\x98\xa9/\xf8'
    framework = '' #tflite, tfRT, tf
    receiver_port = 1234 #port to receive replies from
    receiver_port_verfier = 1235

    input_size = 416
    quality = 65


    target_ip = '192.168.178.23' #contractor ip to receive replies from
    target_ip_verifier = '192.168.178.36' #verifier ip to receive replies from
    #target_port = '5555'

    moving_average_points = 50

    #  '192.168.178.34'
    #port_receiving = 1234
    maxmium_number_of_frames_ahead = 20 #if the frame delay of a contractor gets too high, the contract gets canceled
    minimum_response_rate = 0 #atleast x% of images have to get a response
    warm_up_time = 1500 #number of frames that vialtion of above QOE criteria are not leading to contract abortion (Can be used for handover)

    sampling_interval = 1  #every x images of output an input gets sent to the verifier
    maxmium_number_of_frames_ahead_verifier = 25 #if verifier does not respond with frame in time, sample is considered missed or lost and new one is generated
    maxmium_number_of_verifier_sample_missed_consecutively = 5 #if too many samples are missed or lost after another, contract gets aborted
    minimum_response_rate_verifier = 0.01 #if too many samples are missed or lost in total, contract gets aborted

    framesync = False #syncs framerate of outsourcer and contractor to save network bandwidth



    
   





class ParticipantData:
    balance_outsourcer = 10000
    balance_contractor = 10000
    balance_verifier = 10000



class Helperfunctions:
    
    def hashContract():
       #contract_as_string = ''
       #for attributes in OutsourceContract:
       #    contract_as_string += str(attributes)
       #    contract_as_string += ';'
       #print(vars(OutsourceContract))
       contractHash = hashlib.sha3_256(str(vars(OutsourceContract)).encode('latin1'))
       return contractHash.hexdigest()

    def hashVerifierContract():
       contractHash = hashlib.sha3_256(str(vars(VerifierContract)).encode('latin1'))
       return contractHash.hexdigest()
         