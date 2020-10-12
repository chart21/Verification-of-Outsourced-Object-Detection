import socket
import select

#adds a message header indicating message length to a message and sends it to specified ip and port

class Responder:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.responder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.responder.connect((hostname, port))

    def respond(self, message):

        message = f'{len(message):<10}' + message
        

        self.responder.send(message.encode('latin1'))

    def closeConnection(self):
        self.responder.closeConnection()
