import socket


headersize = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('192.168.178.34', 1234))
s.listen(5)
clientsocket, address = s.accept()

r = receiverLogicThreaded.receiver()


while True:
    # now our endpoint knows about the OTHER endpoint.
    

    full_msg = ''
    new_msg = True
    while True:
        msg = clientsocket.recv(1000)
        if new_msg:
            #print("new msg len:",msg[:headersize])
            msglen = int(msg[:headersize])
            new_msg = False

        #print(f"full message length: {msglen}")

        full_msg += msg.decode("utf-8")

        #print(len(full_msg))


        if len(full_msg)-headersize == msglen:
            #print("full msg recvd")
            print(full_msg[headersize:])
            new_msg = True