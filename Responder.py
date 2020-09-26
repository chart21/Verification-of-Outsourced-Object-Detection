import socket
import select


#header_Length = 10
#ip = '192.168.178.34'
#port = 1234


class Responder:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.responder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.responder.connect((hostname, port))

    def respond(self, message):

        message = f'{len(message):<10}' + message
        #print(message)

        self.responder.send(message.encode('latin1'))

    def closeConnection(self):
        self.responder.closeConnection()
