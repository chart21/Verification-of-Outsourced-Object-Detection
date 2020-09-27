import socket
import threading
import time


class Receiver:


    #def __init__(self, hostname, port):
    def __init__(self):
        
        
        
        
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()
        

        

    def _run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('192.168.178.34', 1234))
        self.s.listen(0)
        self.clientsocket, self.address = self.s.accept()
        self.headersize = 10
        
        while True:
            # now our endpoint knows about the OTHER endpoint.
            
            
            full_msg = ''
            new_msg = True
            while True:
                
                msg = self.clientsocket.recv(20)
                if new_msg:
                    #print("new msg len:",msg[:headersize])
                    msglen = int(msg[:self.headersize])
                    new_msg = False

                #print(f"full message length: {msglen}")

                full_msg += msg.decode("utf-8")

                #print(len(full_msg))

                print(len(full_msg)-self.headersize)
                if len(full_msg)-self.headersize == msglen:
                    #print("full msg recvd")
                    print(full_msg[self.headersize:])
                    
                    new_msg = True
               
            