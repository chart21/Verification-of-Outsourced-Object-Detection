#Class for setting all kinds of differents parameters
import hashlib

# The outsource contract defines the agreement between outsourcer and contractor and has to be set identically for both parties
# If this machine is a verifier, this class is a dummy class and parameters inside are never accessed
class OutsourceContract:
    contract_uid = 0 #contracts have to have a unique id to esnure that each contract hash is unique
    public_key_outsourcer = b'e\x0fy\xfd\xe6\x16\x1f\xe0\x16B\xf2\xdb\x1d\x7f\xc9\xbcLCo\xa7\xa6c\x17\xbf\x8fo\xc8[\x07|bL'
    public_key_contractor = b'\xe9\x919rce\xc9\x1a\xcfJ}\xa3\xee\x17q\x19\xbd\x0eu\xf4\xe0\xd5\x8a<\xc0\x81\x0c\xdbD\xf5;G'
    reward_per_image = 0
    deposit_outsourcer = 0  #deposit of outsourcer to ensure paying fine and reward is possible
    deposit_contractor = 0 #deposit of contractor to ensure paying fine
    fine_outsourcer = 0  #fine if a party is detected cheating
    fine_contractor = 0
    model = 'yolov4' #model to use, possible choices are yolov4, yolov3, Edge TPU uses mobilenet SSD v2 automatically instead
    tiny = True #whether to use tiny weigths for higher performance
    merkle_tree_interval = 128 # 0: Do not use Merkle Tree but sing every output image, >0: Specifies the intervals at wich a Merkle Tree root is signed and sent
    criteria = 'all'   #Specifies if all outputs should be sent back or only outputs that fulfill a certain criteria (e.g certain event happens), criterias should be combined with Merkle Trees to maximize efficiency
    deposit_verfier = 10000000 #verifier details are also set in outsource contract because the contractor creates a list of all available verifier that meet requirements of the outsourcer  
    fine_verifier = 500000
    reward_per_image_verifier = 1


# The outsourcre contract defines the agreement between outsourcer and verifier and has to be set identically for both parties
# If this machine is a contractor, this class is a dummy class and parameters inside are never accessed
class VerifierContract:
    contract_uid = 0 #contracts have to have a unique id to esnure that each contract hash is unique
    public_key_outsourcer = b'e\x0fy\xfd\xe6\x16\x1f\xe0\x16B\xf2\xdb\x1d\x7f\xc9\xbcLCo\xa7\xa6c\x17\xbf\x8fo\xc8[\x07|bL'
    public_key_verifier = b'\xe9\x919rce\xc9\x1a\xcfJ}\xa3\xee\x17q\x19\xbd\x0eu\xf4\xe0\xd5\x8a<\xc0\x81\x0c\xdbD\xf5;G'

    deposit_verfier = 10000000
    fine_verifier = 500000
    reward_per_image_verifier = 1
    
    model = 'yolov4' #model to use, possible choices are yolov4, yolov3
    tiny = True #whether to use tiny weigths for higher performance



  




# class for setting non-contract-related information
class Parameters:
    is_contractor = True #if this machine should act as a verifier or a contractor
    
    private_key_self = b'b\xc8\x8c\xa4\xd5\x82\x18cU\xfa\xdb\x0cg"\x06K\xa7\x01@\x9a\xf7\xa5Yn\x1b>|\x9a\xb6\x02\xaf&' #private Key of contractor
    sendingPort = 1234 #Port to send responses to
    port_outsourcer = 5555 #Port to listen to images


    ip_outsourcer = "192.168.178.34" #Ip address of the outsourcer
    framework = '' #tflite, tfRT, tf
    minimum_receive_rate_from_contractor = 0.9 #outsourcer has to reecive and acknowledge atleast x% of resonses. Otherwise contract is aborted.
    
    # Yolo specific parameters

    framework = ''
    tiny = True
    input_size = 416
    iou = 0.45
    score = 0.5
    weights = './checkpoints/yolov4-tiny-416'
    count = False
    dont_show = False
    info = True
    crop = False

    # Edge TPU specific parameters

    edgeTPU_model_path = 'models_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
    edgeTPU_label_Path = 'labels_edgetpu/coco_labels.txt'
    EdgeTPU_confidence_level = 0.3

    
# Dummy class
class ParticipantData:
    balance_outsourcer = 10000000000000000000
    balance_contractor = 10000000000000000000
    balance_verifier = 100000000000000000000


#Helper class to calculate contract hashes with SHA3-256
class Helperfunctions:
    
    def hashContract():
        
       contractHash = hashlib.sha3_256(str(vars(OutsourceContract)).encode('latin1'))
       return contractHash.hexdigest()

    def hashVerifierContract():
       contractHash = hashlib.sha3_256(str(vars(VerifierContract)).encode('latin1'))
       return contractHash.hexdigest()   

