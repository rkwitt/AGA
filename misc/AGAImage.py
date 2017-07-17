import os
import numpy as np

class AGAImage:
    
    def __init__(self):
        self._relPathToImage = None  # Path to image
        self._nObjects = 0           # Number of all objects in the image
        self._gtBB2D = None          # 2D bounding boxes of all objects in the image
        self._gtBB3DBasis = None     # 3D bounding boxes of all objects in the image    
        self._gtBB3DCentroid = None  # 3D bounding boxes of all objects in the image    
        self._detBB2D = None         # 2D bounding boxes of all objects detected by object detector
        self._labels = []            # Object class names for all objects in the image
       

    def get_label(self, i=None):
        if i is None:
            return self._labels
        if i>=0 and i<self._nObjects:
            return self._labels[i]


    def get_image_path(self):
        return self._relPathToImage


    def get_gtBB2D(self, i=None):
        if i is None:
            return self._gtBB2D
        if i>=0 and i<self._nObjects:
            return self._gtBB2D[i,:]
    

    def get_gtBB3DCentroid(self, i=None):
        if i is None:
            return self._gtBB3DCentroid
        if i>=0 and i<self._nObjects:
            return self._gtBB3DCentroid[i,:]
    
    
    def get_gtBB3DBasis(self, i=None):
        if i is None:
            return self._gtBB3DBasis
        if i>=0 and i<self._nObjects:
            return self._gtBB3DBasis[i,:,:]