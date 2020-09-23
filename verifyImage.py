
from ecdsa import VerifyingKey


class Verfier:
    def __init__(self):
        self.vk = b'Y\xf8D\xe6o\xf9MZZh\x9e\xcb\xe0b\xb7h\xdb\\\xd7\x80\xd2S\xf5\x81\x92\xe8\x109r*U\xebT\x95\x0c\xf2\xf4(\x13%\x83\xb8\xfa;\xf04\xd3\xfb'
        self.vk = VerifyingKey.from_string(self.vk)
        self.vk.precompute()

    def verify(self, name, compressed) :
        return self.vk.verify(bytes(name), compressed)