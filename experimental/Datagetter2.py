import sys

class Datagetter2:
    def __init__(self):
        #self._data_ready = threading.Event()
        self._data_ready = False
        self._data = ''

    def get_data(self):
        #flag = self._data_ready.wait(timeout=timeout)
        if self._data_ready == True:
            self._data_ready = False
            return self._data
        else:
            return -1
        #if waited :
            #print('Waited', (time.perf_counter() - a)*1000)

        self._data_ready.clear()
        
    def setData(self, data):
        self._data = data
        self._data_ready = True
