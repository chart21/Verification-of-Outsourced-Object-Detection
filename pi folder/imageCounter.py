class ImageCounter:

    # def __init__(self, hostname, port):
    def __init__(self, maximum_number_of_frames_ahead):
        self._input_counter = 0
        self._output_counter = 0
        self._frames_ahead = -1
        self._outputs_received =[] 
        self._new_output_frame = False
        self._number_of_outputs_received = 0
        self._maximum_number_of_frames_ahead = maximum_number_of_frames_ahead


    def clearOutputs(self):
        self._outputs_received =[]

    def getInputCounter(self):
        return self._input_counter


    def getNumberofOutputsReceived(self):
        return self._number_of_outputs_received

    def getOutputCounter(self):
        return self._output_counter

    def setOutputCounter(self, counter):
        self._output_counter = counter
        self._frames_ahead = self._input_counter - counter 
        self._new_output_frame = True
        self._number_of_outputs_received += 1

    def increaseInputCounter(self):
        self._input_counter += 1     

    def addOutputReceived(self, outputCount):
        self._outputs_received.append(outputCount)   

    def getFramesAhead(self):
        if self._new_output_frame :
           self._new_output_frame = False
           return self._frames_ahead            
        #else:
        #    if 
        #    return -1
        if self._maximum_number_of_frames_ahead < self._input_counter - self._output_counter:
            return self._input_counter - self._output_counter

        return -1

    

        

