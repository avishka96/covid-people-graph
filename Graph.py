import json, os
import cv2
import numpy as np
from collections import defaultdict

from Node_Person import Person
from suren.util import eprint, stop, progress, Json

# from sklearn.cluster import SpectralClustering

try:
    # import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

except ImportError as e:
    eprint(e)


# SHOW = False  # No idea if this would work when importing @all...maybe call as function?


# Graph visualization packages

class Graph:
    # Static method does not depend on objects created
    @staticmethod
    def plot_import():
        # Why this again?
        try:
            # import networkx
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            return None
        except ImportError as e:
            print(e)
            # SHOW = False  # No idea if this would work when importing @all...maybe call as function?
            return e

    def __init__(self, time_series_length=None, save_name=None):
        """
		:param timeSeriesLength: Number of frames
		"""
        self.time_series_length = time_series_length
        # self.time_series_length = 10
        '''
			Gihan hardcoded to 100 for debug purposes.
		'''
        # Initially following conditions are assumed
        self.n_nodes = 0
        self.n_person = 0
        # Array with n_nodes number of Person Objects
        self.nodes = []

        # State dict
        self.state = {
            "people": 0,  # 1 - people bbox, 2 - tracking id
            "handshake": 0,  # 1 - hs only, 2 - with tracking id, 3 - person info
            "cluster": 0,  # 1 - has cluster info
            "mask": 0,
            "floor": 0,
            # 0b001 - X, Y points only, 0b010 - Interpolation, 0b100 - Projected Floor maps generated with X,Y
            "threat": 0
        }

        self.saveGraphFileName = save_name

        self.BIG_BANG = 0  # HUH -_-
        self.threatLevel = None

        self.PROJECTED_SPACE_H = 1000
        self.PROJECTED_SPACE_W = 1000
        # Projected space coordinates
        self.DEST = [[0, self.PROJECTED_SPACE_H], [self.PROJECTED_SPACE_W, self.PROJECTED_SPACE_H],
                     [self.PROJECTED_SPACE_W, 0], [0, 0]]

        self.projectedFloorMapNTXY = None
        self.REFERENCE_POINTS = None

        self.pairD = None
        self.pairI = None
        self.pairM = None
        self.pairG = None
        self.pairT = None
        self.frameThreatLevel = None

    # __repr__ is a special method used to visualize objects made of a class meaningfully.
    # By calling repr() method, the object could be returned in a string format
    def __repr__(self):
        rep = "Created Graph object with nodes = %d for frames = %d. Param example:\n" % (
            self.n_nodes, self.time_series_length)
        for p in self.nodes[0].params:
            rep += "\t" + str(p) + " " + str(self.nodes[0].params[p]) + "\n"
        return rep

    def project(self, x, y):
        projected = np.dot(self.transMatrix, np.array([x, y, 1]))
        projected = [projected[0] / projected[2], projected[1] / projected[2]]
        return projected

    def get_plot_lim(self, sc_x=None, sc_y=None, hw: tuple = None):

        # LIM based on ref box
        # How y_max, y_min taken??
        x_min, x_max = np.min(self.DEST[:, 0], axis=None), np.max(self.DEST[:, 0], axis=None)
        y_min, y_max = np.min(-1 * self.DEST[:, 1], axis=None), np.max(-1 * self.DEST[:, 1], axis=None)

        x_diff = x_max - x_min
        y_diff = y_max - y_min
        r = .05

        x_min -= x_diff * r
        x_max += x_diff * r  # Changed to x_diff
        y_min -= y_diff * r
        y_max += y_diff * r

        x_lim = [x_min, x_max]
        y_lim = [y_min, y_max]

        # LIM based on plot points
        if sc_x is not None and sc_y is not None:
            x_min = np.nanmin(sc_x, axis=None)
            x_max = np.nanmax(sc_x, axis=None)

            y_min = np.nanmin(sc_y, axis=None)
            y_max = np.nanmax(sc_y, axis=None)

            x_diff = x_max - x_min
            y_diff = y_max - y_min
            r = .05

            x_min -= x_diff * r
            x_max += x_diff * r  # Changed to x_diff

            y_min -= y_diff * r
            y_max += y_diff * r

            # Updating x_lim and y_lim with min and max values
            x_lim[0] = min(x_lim[0], x_min)
            x_lim[1] = max(x_lim[1], x_max)
            y_lim[0] = min(y_lim[0], y_min)
            y_lim[1] = max(y_lim[1], y_max)

        # LIM based on video size
        if hw is not None:
            raise NotImplementedError

            # @GIHAN. TODO : Put 4 endpoints here
            print(hw)
            # x2, y2 = self.project(hw[1], hw[0])

            x_diff = x_max - x_min
            y_diff = y_max - y_min
            r = .01

            x_min -= x_diff * r
            x_max += x_diff * r  # Changed to x_diff

            y_min -= y_diff * r
            y_max += y_diff * r

            x_lim[0] = min(x_lim[0], x_min)
            x_lim[1] = max(x_lim[1], x_max)
            y_lim[0] = min(y_lim[0], y_min)
            y_lim[1] = max(y_lim[1], y_max)

        return x_lim, y_lim

    def get_points_t(self, t):
        scx_det, scy_det, id_det = [], [], []
        scx_interp, scy_interp, id_interp = [], [], []
        # Default dict in python: Does not raise a key error though not present
        line_t = defaultdict(list)
        for n, p in enumerate(self.nodes):
            # initParams = {"id": idx, "xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max, "detection": detected})
            p_id = p.params["id"]
            # p_x = p.params["X"][t]
            # p_y = p.params["Y"][t]

            # Understand the params dictionary
            if p.params["detection"][t]:  # If detections is true
                p_x, p_y = p.params["X_project"][t], p.params["Y_project"][t]

                scx_det.append(p_x)
                # Why append -p_y?
                scy_det.append(-p_y)
                id_det.append(p_id)

                if p.params["handshake"][t]['person'] is not None:  # If handshake detected
                    n1, n2 = sorted([n, p.params["handshake"][t]['person']])
                    line_t["%d_%d" % (n1, n2)].append([p_x, p_y])

            # See what is interpolated?
            if p.params["interpolated"][t]:
                p_x, p_y = p.params["X_project"][t], p.params["Y_project"][t]

                scx_interp.append(p_x)
                scy_interp.append(-p_y)
                id_interp.append(p_id)

        line_t = np.array([line_t[l] for l in line_t])
        if len(line_t) > 0:
            line_t = line_t.transpose((0, 2, 1))

        id_det = np.array(id_det, dtype=int)
        id_interp = np.array(id_interp, dtype=int)

        return scx_det, scy_det, id_det, line_t, scx_interp, scy_interp, id_interp

    def get_scatter_points(self):
        sc_x = []
        sc_y = []
        for t in range(self.time_series_length):
            sc_tx, sc_ty = [], []
            for p in self.nodes:
                p_x = p.params["X"][t]
                p_y = p.params["Y"][t]

                if p.params["detection"][t]:
                    # Call project method
                    p_x, p_y = self.project(p_x, p_y)

                    # pos[n] = (p_x, p_y)
                    sc_tx.append(p_x)
                    sc_ty.append(-p_y)
                else:

                    sc_tx.append(None)
                    sc_ty.append(None)

            sc_x.append(sc_tx)
            sc_y.append(sc_ty)
        sc_x = np.array(sc_x, dtype=float).transpose()
        sc_y = np.array(sc_y, dtype=float).transpose()

        return sc_x, sc_y

    def get_plot_points_all(self):

        # assert self.state["floor"] >= 1, "Need X, Y points to plot graph"     # @suren : TODO

        # ori_x = []
        # ori_y = []

        sc_x = []
        sc_y = []
        lines = []
        for t in range(self.time_series_length):
            # pos = {}
            # ori_tx, ori_ty = [], []
            sc_tx, sc_ty = [], []
            line_t = defaultdict(list)
            for n, p in enumerate(self.nodes):
                p_x = p.params["X"][t]
                p_y = p.params["Y"][t]

                if p.params["detection"][t]:
                    # ori_tx.append(p_x)
                    # ori_ty.append(p_y)
                    p_x, p_y = self.project(p_x, p_y)

                    # pos[n] = (p_x, p_y)
                    sc_tx.append(p_x)
                    sc_ty.append(-p_y)

                    if p.params["handshake"][t]['person'] is not None:
                        n1, n2 = sorted([n, p.params["handshake"][t]['person']])
                        line_t["%d_%d" % (n1, n2)].append([p_x, p_y])
                    # print(t, n1, n2, p_x, p_y)
                else:
                    # ori_tx.append(None)
                    # ori_ty.append(None)

                    sc_tx.append(None)
                    sc_ty.append(None)

            sc_x.append(sc_tx)
            sc_y.append(sc_ty)

            # ori_x.append(ori_tx)
            # ori_y.append(ori_ty)

            # print("XXX", line_t)
            # @suren : find a better way to implement variable size array
            try:
                line_t = np.array([line_t[l] for l in line_t]).transpose((0, 2, 1))
            except ValueError:
                line_t = []
            lines.append(line_t)
        sc_x = np.array(sc_x, dtype=float).transpose()
        sc_y = np.array(sc_y, dtype=float).transpose()

        # ori_x = np.array(ori_x, dtype=float).transpose()
        # ori_y = np.array(ori_y, dtype=float).transpose()

        return sc_x, sc_y, lines

    def get_cmap(self, n: int = None, show=False):
        if n is None:
            n = self.n_nodes

        colors = cm.hsv(np.linspace(0, .8, n))
        window = 10
        col_arr = np.ones((window, 4))
        # np.power(arr1,arr2): Raise an arr into a given power through another arr
        # np.arange(start=0,stop,step=1): Create evenly spaced array from [start,stop) in step steps
        col_arr[:, -1] = np.power(.8, np.arange(window))[::-1]
        # Convert arr to title font
        arr1 = np.tile(colors, (window, 1, 1)).transpose((1, 0, 2))
        # print(colors.shape, arr1.shape)
        arr2 = np.tile(col_arr, (n, 1, 1))
        # print(col_arr.shape, arr2.shape)
        cmap = arr1 * arr2
        # print(arr1[1, :, :], arr2[1, :, :])
        # print(colors)
        # stop()
        if show:
            x = np.tile(np.arange(cmap.shape[0]), (cmap.shape[1], 1))
            y = np.tile(np.arange(cmap.shape[1]), (cmap.shape[0], 1)).transpose()
            # print(x)
            # print(y)
            plt.figure()
            plt.title("Colour map (Close to continue)")
            plt.scatter(x.flatten(), y.flatten(), color=np.reshape(cmap, (-1, 4), order='F'))
            plt.show()
        return cmap

    # Plot with moving points... don't use this. Visualizer has a better implementation.
    def plot(self, window=10, show_cmap=True):
        if Graph.plot_import() is not None:
            eprint("Package not installed", Graph.plot_import())
            return

        # plt.figure()
        # colour map -> call get_cmap()
        cmap = self.get_cmap(show=show_cmap)
        # scatter x, y and lines
        sc_x, sc_y, lines = self.get_plot_points_all()
        # print(sc_x.shape, sc_y.shape, cmap.shape)
        # PLOT
        xlim, ylim = self.get_plot_lim(sc_x, sc_y)

        fig = plt.figure()
        # Limits of the figure (xlim = [xmin,xmax], ylim = [ymin,ymax])
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        ax = plt.gca()
        plt.ion()
        for t in range(self.time_series_length):
            sc_x_ = sc_x[:, max(t + 1 - window, 0):t + 1]
            sc_y_ = sc_y[:, max(t + 1 - window, 0):t + 1]
            cmap_ = cmap[:, max(0, window - (t + 1)):, :]

            # print(sc_x_.shape, sc_y_.shape, cmap_.shape)

            ax.scatter(sc_x_.flatten(), sc_y_.flatten(), color=np.reshape(cmap_, (-1, 4), order='C'))

            for l in lines[t]:
                ax.plot(l[0], l[1])
                plt.pause(.3)


            else:
                plt.pause(.1)

            ax.clear()
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

            if (t + 1) % 20 == 0:
                progress(t + 1, self.time_series_length, "drawing graph")

    # plt.show(block=True)

    def get_nxt_id(self):
        # nodes starting from 0. Thus, len(self.nodes) will give the next id in line
        return len(self.nodes)

    def add_person(self, p: Person = None):
        # Create a person object from Person class if p is none
        if p is None:
            p = Person(time_series_length=self.time_series_length, idx=self.get_nxt_id())
        # Assign next id in line to p object
        elif p.idx is None:
            p.idx = self.get_nxt_id()

        # Insert p object as a node in graph
        self.nodes.append(p)
        self.n_person += 1
        self.n_nodes = len(self.nodes)

        return p

    # def addNode(self,time):
    # 	print("GRAPH: adding (person) node")
    # 	self.nodes.append(Person())
    # 	return len(self.nodes)-1

    # Obtain node from a particular id. Note: Each node is a person. Thus. obtaining node is equivalent to obtaining a
    # person
    def getNode(self, idx):
        return self.nodes[idx]

    def make_jsonable(self, data):
        for node in data["nodes"]:
            for param in node:
                print(param, node[param])
                if param == "handshake":
                    for t in range(self.time_series_length):
                        print(type(node[param][t]['person']))
                        print(type(node[param][t]['confidence']))

                else:
                    for t in range(self.time_series_length):
                        print(type(node[param][t]))
        # print(type(node["handshake"]))

    def saveToFile(self, file_name=None):
        if file_name is None:
            # saveGraphFileName is an attribute which has arguments depending on graph object initialization
            file_name = self.saveGraphFileName

        # "state" key:refer self.state dictionary
        # "nodes" key:n is each node (i.e. person) and n.params is parameters of n (i.e. person)
        # Understand the structure of data dictionary
        data = {
            "N": self.n_nodes,
            "frames": self.time_series_length,
            "state": self.state,
            "nodes": [n.params for n in self.nodes]
        }

        # is_jsonable in suren.util
        Json.is_jsonable(data)  # Delete this later @suren

        # Create json file
        js = Json(file_name)
        # write data to json file
        js.write(data)

        # write success message
        print("Finished writing all nodes to {}".format(file_name))

    def getCameraInfoFromJson(self, fileName):
        with open(fileName) as json_file:
            data = json.load(json_file)

        self.REFERENCE_POINTS = data["reference_points"]
        # Converting float data points to 32 bit
        self.REFERENCE_POINTS = np.float32(self.REFERENCE_POINTS)

        self.DEST = np.float32(self.DEST)
        # self.DEST includes the boundaries of the mapped 2D space
        # opencv's getPerspectiveTransform generates a transformation matrix to map reference points of the camera onto
        # destination points
        self.transMatrix = cv2.getPerspectiveTransform(self.REFERENCE_POINTS, self.DEST)

        self.GROUP_DIST_THRESH = data["group_radius_threshold"]
        self.GROUP_TIME_THRESH = data["group_time_threshold"]

        self.DISTANCE_TAU = data["distance_tau"]

    # Sample json file having camera information
    # {
    #	"reference_points": [[115, 1000], [1640, 1000], [1320, 300], [670, 290]],
    #	"group_radius_threshold": 100.0,
    #	"group_time_threshold": 0.2,
    #	"distance_tau": 400.0
    # }

    def init_from_json(self, file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)

        try:
            N = data["N"]
            # Raise assertion error if N not equal to no. of nodes
            assert len(data["nodes"]) == N, "No of nodes not equal to N"

        except Exception as e:
            eprint(e)
            N = len(data["nodes"])

        # N = 0 means no person. Thus no nodes
        if N == 0:
            eprint("No nodes :(")
            return

        try:
            time_series_length = data["frames"]
            # Raise assertion error if len(data["nodes"][0]["detection"]) not equal to no of frames
            # But what is data["nodes"][0]["detection"]?
            assert len(data["nodes"][0]["detection"]) == time_series_length, "Time series length not equal"

        except Exception as e:
            eprint(e)
            time_series_length = len(data["nodes"][0]["detection"])

        # state dict = "state" key of data dict
        self.state = data["state"]

        if self.time_series_length is None:
            self.time_series_length = time_series_length

        for n in range(N):
            # adding person for each node
            p = self.add_person()
            # setting parameters of each person from dict (json) data
            # setParamsFromDict() in node.py
            p.setParamsFromDict(data["nodes"][n])

    # def calculate_standing_locations(self):
    # 	for n in self.nodes:
    # 		n.calculate_standing_locations()

    # def interpolate_undetected_timestamps(self):
    # 	for n in self.nodes:
    # 		n.interpolate_undetected_timestamps()

    def generateFloorMap(self, verbose=False, debug=False):

        # Understand the structure of self.state
        # What is self.state["people"]?
        assert self.state["people"] >= 2, "Floor map cannot be generated without people bbox"

        # what does self.state["floor"] means?
        if self.state["floor"] < 1:
            for n in self.nodes:
                # calculate_standing_location() is from Node_Person.py
                # middle of the bbox
                n.calculate_standing_locations()
            # self.state["floor"] = 1     # @suren : TODO

            # if True:
            for n in self.nodes:
                # interpolate_undetected() from Node_Person.py
                n.interpolate_undetected(debug=debug)
            # self.state["floor"] |= 1 << 1     # @suren : TODO

            # if True:
            for n in self.nodes:
                # project_standing_locations fron Node_Person.py
                # Projects the actual standing locations through perspective transform to the 2D space
                n.project_standing_location(self.transMatrix)

        self.state["floor"] = 1

        # Floor map N x T with X and Y points.
        # self.projectedFloorMapNTXY is a num node(row) * num frames(col) * 2(depth) zeros matrix
        self.projectedFloorMapNTXY = np.zeros((self.n_nodes, self.time_series_length, 2), dtype=np.float32)

        for n, node in enumerate(self.nodes):
            # nth row, all cols, 0th depth
            self.projectedFloorMapNTXY[n, :, 0] = node.params["X_project"]
            # nth row, all cols, 1st depth
            self.projectedFloorMapNTXY[n, :, 1] = node.params["Y_project"]

        # X = self.nodes[n].params["X"]
        # Y = self.nodes[n].params["Y"]
        # for t in range(self.time_series_length):
        #
        # 	projected=self.project(X[t], Y[t])
        # 	self.projectedFloorMapNTXY[n,t,0]=projected[0]
        # 	self.projectedFloorMapNTXY[n,t,1]=projected[1]
        #
        # np.testing.assert_almost_equal(self.projectedFloorMapNTXY[n, :, 0], node.params["X_project"], decimal = 5)
        # np.testing.assert_almost_equal(self.projectedFloorMapNTXY[n, :, 1], node.params["Y_project"], decimal =5)
        #
        # raise NotImplementedError

        # self.state["floor"] |= 1 << 2    # @suren : TODO

        if verbose:
            print("Finished creating floormap {} ".format(self.projectedFloorMapNTXY.shape))

    # Group Identification Algorithm
    def findClusters(self, METHOD="NAIVE", debug=False, verbose=False):

        # self.groupProbability is a num nodes(row) * num nodes(col) * num frames(depth) matrix
        # Vertical cross section is a square
        self.groupProbability = np.zeros((self.n_nodes, self.n_nodes, self.time_series_length), np.float32)

        # There is a lot for me to do on this array.
        # Similar to above
        self.pairDetectionProbability = np.zeros((self.n_nodes, self.n_nodes, self.time_series_length), np.float32)

        if METHOD == "NAIVE":
            for p1 in range(self.n_nodes):
                for p2 in range(self.n_nodes):
                    if p1 <= p2:  # Filling the lower triangle only. Cauz, symmetrical
                        if not self.nodes[p1].params["neverDetected"] and not self.nodes[p2].params["neverDetected"]:
                            t1 = max(self.nodes[p1].params["detectionStartT"], self.nodes[p2].params["detectionStartT"])

                            # Isn't it min(...) for t2? Cauz, both p1 and p2 must be detected?
                            t2 = max(self.nodes[p1].params["detectionEndTExclusive"],
                                     self.nodes[p2].params["detectionEndTExclusive"])

                            # From t1 to t2, p1 and p2 pair could be detected. Thus, Prob = 1
                            self.pairDetectionProbability[p1, p2, t1:t2] = 1.00

                        tempDistDeleteThisVariableLater = []

                        for t in range(self.time_series_length):
                            # Obtaining the Euclidean distance between p1 and p2
                            dist = np.sqrt(np.sum(
                                np.power(self.projectedFloorMapNTXY[p1, t, :] - self.projectedFloorMapNTXY[p2, t, :],
                                         2)))
                            tempDistDeleteThisVariableLater.append(dist)

                            # self.GROUP_DIST_THRESH obtained from camera info json file
                            if dist < self.GROUP_DIST_THRESH:
                                # At frame = t, p1 and p2 is within a groupable distance. Cauz, Prob = 1
                                self.groupProbability[p1, p2, t] = 1.0
                            else:
                                self.groupProbability[p1, p2, t] = 0.0

                        if debug:
                            # print("Dist between {} and  {}:".format(p1,p2),tempDistDeleteThisVariableLater)
                            print("Dist between {} and  {}:".format(p1, p2), tempDistDeleteThisVariableLater[t1:t2])
                    else:
                        # Cauz, symmetric
                        self.groupProbability[p1, p2, :] = self.groupProbability[p2, p1, :]
                        self.pairDetectionProbability[p1, p2, :] = self.pairDetectionProbability[p2, p1, :]

        # Spectral method was not implemented
        if METHOD == "SPECTRAL":
            print("THIS METHOD WAS REMOVED!!!")
            """ <<<< end """

        # (a,b,c) are temporary variables
        # a is num nodes * num nodes * num frames matrix.
        # a is element wise product of self.groupProb and self.pairDetProb a matrix has the possibilities
        # of any two person, p1 and p2 being detected as a pair (in frames together) and being in a group(dist < thresh)
        a = self.groupProbability * self.pairDetectionProbability

        # Taking sum along depth (time axis).
        # That is any person p1 total possibilities of being detected as a pair and being in a group,
        # with every person
        b = np.sum(a, -1)

        # Taking sum along depth (time axis).
        # That is total possibilities of any person p1 could be in a group with every person
        # Logic : Any two person once detected as a pair has the possibility of beign in a group
        c = np.sum(self.pairDetectionProbability, -1)

        # Applying definition of probability
        self.groupProbability = b / c

        if verbose:
            print("Group probability", self.groupProbability)

        # Valid only if > thresh
        self.groupProbability = self.groupProbability > self.GROUP_TIME_THRESH

        if verbose:
            print("Group probability (bianry)", self.groupProbability)

        self.state["cluster"] = 1     # @suren : TODO
        # Avishka uncommented this

    def calculateThreatLevel(self, debug=False):
        # P = num persons/nodes
        P = len(self.nodes)
        # T = num frames
        T = self.time_series_length

        # All below are num frames(row) * num person(col) * num person(depth) matrices
        self.pairD = np.zeros((T, P, P), dtype=np.float32)
        self.pairI = np.zeros((T, P, P), dtype=np.float32)
        self.pairM = np.zeros((T, P, P), dtype=np.float32)
        self.pairG = np.zeros((T, P, P), dtype=np.float32)

        # threat level for each time frame, for each person pair
        self.pairT = np.zeros((T, P, P), dtype=np.float32)

        # Threat level is an array given for every frame
        self.frameThreatLevel = np.zeros((T), dtype=np.float32)

        # t is any time frame
        for t in range(T):
            threatLevel = 0.0

            # p1 is any peson
            for p1 in range(P):
                interact = self.nodes[p1].params["handshake"][t]

                # p2 is any person
                for p2 in range(P):
                    # No point of inspecting p1 = p2 case
                    if p1 != p2:
                        # Calculating the euclidean norm between projected distances of p1 and p2
                        d = np.linalg.norm(self.projectedFloorMapNTXY[p1, t, :] - self.projectedFloorMapNTXY[p2, t, :])
                        # Wtf is this?
                        d = np.exp(-1.0 * d / self.DISTANCE_TAU)
                        # if p1 is interacting with p2, i=1, else i=0
                        i = 1 if interact["person"] == p2 else 0  # get from graph self.nodes @Jameel
                        # Mask not included in analysis
                        m = 0.0  # get from graph self.nodes @Suren
                        # g denotes probability that p1,p2 pair could be in a group
                        g = self.groupProbability[p1, p2]

                        # Distance
                        self.pairD[t, p1, p2] = d * self.pairDetectionProbability[p1, p2, t]
                        # Handshake
                        self.pairI[t, p1, p2] = i * self.pairDetectionProbability[p1, p2, t]
                        # Mask
                        self.pairM[t, p1, p2] = m * self.pairDetectionProbability[p1, p2, t]
                        # Group
                        self.pairG[t, p1, p2] = g * self.pairDetectionProbability[p1, p2, t]

                        # Epsilon values
                        EPS_m = 2.0
                        EPS_g = 2.0

                        # Equation to get the threat level for pair p1,p2
                        threatOfPair = (d + i) * (EPS_m - m) * (EPS_g - g) * self.pairDetectionProbability[p1, p2, t]
                        # threat level for a particular frame t=t is given by sum of all pairwise threat levels
                        threatLevel += threatOfPair

                        # threat level for time frame = t, for persons p1,p2
                        self.pairT[t, p1, p2] = threatOfPair
            # threat level for a particular frame
            self.frameThreatLevel[t] = threatLevel
        print("Finished calculating threat level")

        #self.state["threat"] = 1     # @suren : TODO
        # Avishka uncommented this

    # See with an example
    def fullyAnalyzeGraph(self):
        self.generateFloorMap()
        self.findClusters()
        self.calculateThreatLevel()

    # Wtf is this?
    def set_ax(self, ax, n):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(i + 1) for i in range(n)])
        ax.set_yticklabels([str(i + 1) for i in range(n)])

    def threat_image_init(self, fig, ax):
        # ax.spines.right.set_visible(False)
        # ax.spines.bottom.set_visible(False)
        # ax.tick_params(bottom=False, labelbottom=False)
        T_max = np.max(self.pairT, axis=None)
        im = ax.matshow(self.pairT[0, :, :], vmin=0, vmax=T_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.clear()
        fig.savefig("./data/output/threat_image_init.jpg")

    def threat_image_save(self, fig, ax, out_name, t):
        T_max = np.max(self.pairT, axis=None)
        ax.matshow(self.pairT[t, :, :], vmin=0, vmax=T_max)
        self.set_ax(ax, self.n_nodes)
        # n, m = self.pairT[t, :, :].shape
        # ax.set_xticklabels([str(i) for i in range(n+1)])
        # ax.set_yticklabels([str(i) for i in range(m+1)])
        fig.savefig("{}T-{:04d}.jpg".format(out_name, t))
        ax.clear()

    def threat_image(self, fig, out_name, t):
        fig.clf()
        ax = fig.add_axes([0, 0, 1, 1])
        im = ax.matshow(self.pairT[t, :, :])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig("{}T-{:04d}.jpg".format(out_name, t))

    def image_init(self, ax, xlim, ylim):
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.clear()
        ax.set_axis_off()

    def image_save(self, fig1, ax1, xlim, ylim, out_dir, t, clear=True):

        fig1.savefig("{}G-{:04d}.jpg".format(out_dir, t))
        if clear:
            ax1.clear()
            ax1.set_xlim(xlim[0], xlim[1])
            ax1.set_ylim(ylim[0], ylim[1])
            ax1.set_axis_off()

    def dimg_init_concat(self, fig, ax):
        vals = {
            "d": (self.pairD[0, :, :], "Distance"),
            "i": (self.pairI[0, :, :], "Interaction"),
            "m": (self.pairM[0, :, :], "Mask"),
            "g": (self.pairG[0, :, :], "Group")
        }

        if len(ax) == 2:
            ax = [ax[i][j] for i in range(2) for j in range(2)]

        for col, ind in zip(ax, vals):
            # col.spines.right.set_visible(False)
            # col.spines.top.set_visible(False)
            col.title.set_text(vals[ind][1])
            im = col.matshow(vals[ind][0], vmin=0, vmax=1)
            divider = make_axes_locatable(col)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        for col in ax:
            col.clear()
        # 	col.spines.right.set_visible(False)
        # 	col.spines.top.set_visible(False)

        fig.savefig("./data/output/dimg_init_concat.png")

    def dimg_init_full(self, fig, ax):
        ax.clear()
        # ax.spines.right.set_visible(False)
        # ax.spines.top.set_visible(False)
        im = ax.matshow(self.pairD[0, :, :], vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.clear()
        fig.savefig("./data/output/dimg_init_full.jpg")

    def dimg_init(self, fig2, ax2, fig4, ax4):
        self.dimg_init_concat(fig2, ax2)
        self.dimg_init_full(fig4, ax4)

    def dimg_save_concat(self, fig, ax, out_name, t):
        vals = {
            "d": (self.pairD[t, :, :], "Distance"),
            "i": (self.pairI[t, :, :], "Interaction"),
            "m": (self.pairM[t, :, :], "Mask"),
            "gr": (self.pairG[t, :, :], "Group")
        }

        if len(ax) == 2:
            ax = [ax[i][j] for i in range(2) for j in range(2)]

        n, m = self.pairD[t, :, :].shape

        for col, ind in zip(ax, vals):
            col.title.set_text(vals[ind][1])
            col.matshow(vals[ind][0], vmin=0, vmax=1)
            self.set_ax(col, self.n_nodes)
        # col.set_xticklabels([str(i) for i in range(n+1)])
        # col.set_yticklabels([str(i) for i in range(m+1)])
        fig.savefig("{}dimg-{:04d}.jpg".format(out_name, t))
        for col in ax:
            col.clear()

    def dimg_save_full(self, fig, ax, out_name, t):
        # if path does not exist, make directory
        if not os.path.exists("{}/figs".format(out_name)):
            os.makedirs("{}/figs".format(out_name))
        vals = {
            "d": (self.pairD[t, :, :], "Distance"),
            "i": (self.pairI[t, :, :], "Interaction"),
            "m": (self.pairM[t, :, :], "Mask"),
            "gr": (self.pairG[t, :, :], "Group")
        }

        n, m = self.pairD[t, :, :].shape
        for ind in vals:
            ax.matshow(vals[ind][0], vmin=0, vmax=1)
            self.set_ax(ax, self.n_nodes)
            # ax.set_xticklabels([str(i) for i in range(n+1)])
            # ax.set_yticklabels([str(i) for i in range(m+1)])
            fig.savefig("{}/figs/{}-{:04d}.jpg".format(out_name, ind, t))
            ax.clear()

    def dimg_save(self, fig2, ax2, fig4, ax4, out_name, t):
        self.dimg_save_concat(fig2, ax2, out_name, t)
        self.dimg_save_full(fig4, ax4, out_name, t)


if __name__ == "__main__":
    g = Graph()
    g.init_from_json('./data/temp_files/vid-01-graph.json')		# Start from yolo

    #g.init_from_json('./data/temp_files/vid-01-graph_handshake.json')  # Start from handshake
    g.getCameraInfoFromJson('./data/camera-orientation/jsons/deee.json')
    g.fullyAnalyzeGraph()

    # print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
    print(g.pairD)
