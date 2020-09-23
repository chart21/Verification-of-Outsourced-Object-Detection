import hmac
import hashlib

#img=mpimg.imread('image_name.png')
secret_key = b"NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
total_params = b"/public/api/ver1/accounts/new?type=binance&name=binance_account&api_key=XXXXXX&secret=YYYYYY"
#signature = hmac.new(secret_key, total_params, hashlib.sha256).hexdigest()

#print("signature = {0}".format(signature))
#hmac.new()

import rsa

(pubkey, privkey) = rsa.newkeys(1024)
print(pubkey, privkey)
#message = 'bla'.encode('utf-8')
message = hashlib.sha3_256('blaaaaa'.encode('utf-8')).hexdigest().encode('utf-8')

print(message)
#encrypt
crypto = rsa.encrypt(message, pubkey)
print(crypto)
#decrypt
message = rsa.decrypt(crypto, privkey)
print(message)

