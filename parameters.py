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









class Parameters:
    ip_outsourcer = "192.168.178.34"
    port_outsourcer = 5555
    private_key_contractor = b'b\xc8\x8c\xa4\xd5\x82\x18cU\xfa\xdb\x0cg"\x06K\xa7\x01@\x9a\xf7\xa5Yn\x1b>|\x9a\xb6\x02\xaf&'
    framework = '' #tflite, tfRT, tf
    sendingPort = 1234

    minimum_receive_rate_from_contractor = 0.9 #contractor has to reecive and acknowledge atleast x% of resonses. Otherwise contract is aborted.
    
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

class ParticipantData:
    balance_outsourcer = 10000
    balance_contractor = 10000
    balance_verifier = 10000



class Helperfunctions:
    
    def hashContract():
        
       contractHash = hashlib.sha3_256(str(vars(OutsourceContract)).encode('latin1'))
       return contractHash.hexdigest()

