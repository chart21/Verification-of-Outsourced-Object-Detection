import sys
import threading
class Datagetter:
    def __init__(self):
        self._data_ready = threading.Event()
        self._data = ''

    def get_data(self, timeout = 15):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Contract aborted: Outsourcer at tcp://{}:{}".format(self.hostname, self.port) + 'timed out. Possible Consquences for Outsourcer: Blacklist, Bad Review')

        #if waited :
            #print('Waited', (time.perf_counter() - a)*1000)

        self._data_ready.clear()
        
    def setData(self, data):
        self._data = data
        self._data_ready.set()
