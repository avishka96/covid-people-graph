from Node import *
import os
import numpy as np


# Inherited from Node class
class Person(Node):

    def __init__(self, isLocation2D=True, initParams=None, time_series_length=None, idx=None):
        """
		@gihan what are these params
		:param isLocation2D: ??? @gihan
		:param time_series_length: Number of time samples
		"""

        super().__init__(initParams=None, time_series_length=time_series_length, idx=idx)

        self.location2D = isLocation2D
        # self.type = 'Person'

        self.init_pos()
        self.init_handshake()

        if initParams is not None:
            self.setParamsFromDict(initParams)

    # for par in self.params:
    # 	print("NODE_PERSON", par, self.params[par])

    # def setInitialLocation(self,X,Y,Z=None):
    # 	self.params["X"][0]=X
    # 	self.params["Y"][0]=Y

    def init_handshake(self):
        # print("Initializing handshake")
        self.params["handshake"] = [{"person": None, "confidence": None, "iou": None, "id": None} for _ in
                                    range(self.time_series_length)]

    # self.addParam("handshake")
    # for t in range(self.time_series_length):	# @gihan : Why are there multiple time series lengths? Shouldn't this be global?
    # 	self.setParam("handshake", t, {"person":None,"confidence":None})

    def init_pos(self):
        self.params["xMin"] = [0 for _ in range(self.time_series_length)]
        self.params["yMin"] = [0 for _ in range(self.time_series_length)]
        self.params["xMax"] = [0 for _ in range(self.time_series_length)]
        self.params["yMax"] = [0 for _ in range(self.time_series_length)]

        self.params["detection"] = [False for _ in range(self.time_series_length)]

    def calculate_standing_locations(self):
        if "X" not in self.params.keys():
            self.addParam("X")
        if "Y" not in self.params.keys():
            self.addParam("Y")

        # pointOnFloorX=(frames[t][pt]["bbox"][0]+frames[t][pt]["bbox"][2])/2
        # pointOnFloorY=frames[t][pt]["bbox"][3]

        # node.setParam("X",t,pointOnFloorX)
        # node.setParam("Y",t,pointOnFloorY)
        # node.setParam("detection",t,True)

        for t in range(self.time_series_length):
            X = int((self.params["xMin"][t] + self.params["xMax"][t]) / 2)
            self.setParam("X", t, X)
            self.setParam("Y", t, self.params["yMax"][t])       # Changed to yMin

    def calculate_detected_time_period(self, debug=False):
        f_name = os.path.basename(__file__)

        if debug:
            print("\t {} [DEBUG]: running person calc".format(f_name))

        startT = 0
        endTExclusive = self.time_series_length

        # print(self.params["detection"])
        for t in range(0, self.time_series_length):
            if self.params["detection"][t] == True:
                break
            startT = t + 1
        for t in range(self.time_series_length - 1, 0, -1): #start->(time_series_length-1), stop->0, step->-1
            if self.params["detection"][t] == True:
                break
            endTExclusive = t

        if startT < endTExclusive:  #This means detected
            self.params["neverDetected"] = False
            self.params["detectionStartT"] = startT
            self.params["detectionEndTExclusive"] = endTExclusive
        else:
            self.params["neverDetected"] = True

    def interpolate_undetected_timestamps(self, debug=False):   #Avi making note : Refer this again !! (02.06.22)
        '''
			Gihan (25/03/2021)
			I am implementing this without thinking straight.
			I will be refining this logic later on.
		'''
        f_name = os.path.basename(__file__)

        self.calculate_standing_locations()
        self.calculate_detected_time_period(debug=debug)
        self.params["interpolated"] = [False for _ in range(self.time_series_length)]

        if not self.params["neverDetected"]:
            t1 = self.params["detectionStartT"]
            while t1 < self.params["detectionEndTExclusive"]:
                if self.params["detection"][t1] == True:
                    t1 += 1
                else:
                    t2 = t1
                    while t2 < self.params["detectionEndTExclusive"]:   #false detection removal
                        if self.params["detection"][t2] == False:
                            t2 += 1
                        else:
                            # @GIHAN : WTF?
                            if debug:
                                print("\t " + f_name + " [DEBUG]: INTERPOLATION (before): ", t1, t2)

                                # print("DEBUG INTERPOLATION: ",self.params["detection"])
                                for a in range(t1 - 2, t2 + 2):
                                    print("\t\t ", a, self.params["detection"][a], self.params["X"][a],
                                          self.params["Y"][a])

                            # Now we know t1----t2(exclusive) are false detections.
                            toFillCount = t2 - t1
                            xStep = (self.params["X"][t2] - self.params["X"][t1 - 1]) / toFillCount
                            yStep = (self.params["Y"][t2] - self.params["Y"][t1 - 1]) / toFillCount
                            for tt in range(toFillCount):
                                self.setParam("X", t1 + tt, self.params["X"][t1 - 1] + tt * xStep)
                                self.setParam("Y", t1 + tt, self.params["Y"][t1 - 1] + tt * yStep)
                                self.setParam("interpolated", t1 + tt, True)

                            if debug:
                                print("\t " + f_name + " [DEBUG]: INTERPOLATION (after): ", t1, t2)

                                # print("DEBUG INTERPOLATION: ",self.params["detection"])
                                for a in range(t1 - 2, t2 + 2):
                                    print("\t\t ", a, self.params["detection"][a], self.params["X"][a],
                                          self.params["Y"][a])

                            t1 = t2
                            break

    def interpolate_undetected(self, debug=False):  #Avi making note : Refer this again !! (02.06.22)
        '''
			Gihan (25/03/2021)
			I am implementing this without thinking straight.
			I will be refining this logic later on.

			@suren updated this
		'''

        f_name = os.path.basename(__file__)

        self.params["interpolated"] = [False for _ in range(self.time_series_length)]

        if "X" not in self.params.keys() or "Y" not in self.params.keys():
            self.calculate_standing_locations()

        det = np.where(self.params["detection"])[0]
        if len(det) > 0:
            start = det[0]
            end = det[-1] + 1
            self.params["neverDetected"] = False
            self.params["detectionStartT"] = int(start)
            self.params["detectionEndTExclusive"] = int(end)

            X = self.params["X"]
            Y = self.params["Y"]

            x_det = np.array(X)[det]
            y_det = np.array(Y)[det]

            x_interp = np.interp(np.arange(start, end), det, x_det)
            y_interp = np.interp(np.arange(start, end), det, y_det)

            X[start:end] = x_interp.tolist()
            Y[start:end] = y_interp.tolist()

            self.params["interpolated"][start:end] = np.invert(self.params["detection"][start:end]).astype(
                'bool').tolist()
            self.params["interpolated"] = self.params["interpolated"]

            self.params["X"] = X
            self.params["Y"] = Y

        else:
            self.params["neverDetected"] = True

    def project_standing_location(self, transMatrix, debug=False):

        XY_mat = np.array([self.params["X"], self.params["Y"], np.ones(self.time_series_length)])
        projected = np.dot(transMatrix, XY_mat)

        projected[0, :] /= projected[2, :]
        projected[1, :] /= projected[2, :]

        self.params["X_project"] = projected[0, :].tolist()
        self.params["Y_project"] = projected[1, :].tolist()
