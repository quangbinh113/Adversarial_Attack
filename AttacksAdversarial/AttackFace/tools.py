import re
import os
import sys
from skimage import io
import numpy as np
from skimage.draw import polygon_perimeter

class FaceBox:
    """
    A class for storing bounding boxes of faces

    Attributes:
        x1: x position of upper-left box corner in pixels 
        y1: y position of upper-left box corner in pixels
        width: box width in pixels
        height: box height in pixels
        quality: if ground truth, this is a TruthBoxQuality object; for found boxes, it's None

    Methods:
        area: returns the area of the bounding box in pixels as a float
        iou: returns the intersection over union between two FaceBox objects
    """
    def __init__(self, x1=0, y1=0, width=0, height=0, quality=None):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.width = int(width)
        self.height = int(height)
        self.quality = quality
    
    def __str__(self):
        return "(x1:{} y1:{} Width:{} Height:{} Quality: {})".format(
            self.x1, 
            self.y1, 
            self.width, 
            self.height, 
            self.quality) 
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def area(self):
        """Area of FaceBox as a float"""
        return float(self.width*self.height)
    
    def iou(self, box):
        """Intersection over union of two FaceBox objects as a float"""
        if (self.width == 0 or self.height == 0 or box.width == 0 or box.height == 0): return 0
        intersect = FaceBox( max(0,self.x1,box.x1), max(0,self.y1,box.y1), 
                max(0,min(self.x1+self.width,box.x1+box.width)-max(0,self.x1,box.x1)), 
                max(0,min(self.y1+self.height,box.y1+box.height)-max(0,self.y1,box.y1))  )
        return max(0,intersect.area() / (self.area() + box.area() - intersect.area()))


class FaceBoxMatch:
    """
    A class for storing a match between two FaceBox objects

    Attributes:
        target_box: the FaceBox you are attempting to match 
        match_box: the FaaceBox matched to the target
    """
    def __init__(self, target_box=FaceBox(), match_box=FaceBox()):
        self.target_box = target_box
        self.match_box = match_box
    
    def __str__(self):
        return "(Target Box:{} Match Box:{} IoU:{})".format(
            self.target_box, 
            self.match_box, 
            self.target_box.iou(self.match_box))
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__


def get_found_boxes(image, detector, upsample=1):
    """
    Get a list of FaceBox objects found by a given detector

    Args:
       image: a numpy array for a given image
       detector: the face detector you want to use
       upsample: optional upsampling value
    Returns:
       found_box_list: list of FaceBox objects found in the image   
    """
    try: found_faces = detector(image, upsample)
    except:
        error_str = ('get_found_boxes detector object failed to find boxes')
        sys.exit(error_str)
    found_box_list = []
    for face in found_faces:
        width = face.right()-face.left()
        height = face.bottom()-face.top()
        if (face.right() > image.shape[1]): width = image.shape[1] - face.left()
        if (face.bottom() > image.shape[0]): height = image.shape[0] - face.bottom()
        found_box_list.append(FaceBox(face.left(), face.top(), width, height))
    return found_box_list


def draw_boxes(image, box_list, color_rgb):
    """
    Draw FaceBoxes of a given color onto image

    Args:
        image: an image for drawing over
        box_list: a list of FaceBox objects to draw
        color_rgb: the color to draw the box, given as an RGB triplet 
    """
    for box in box_list:
        right_edge = box.x1+box.width-1
        if (right_edge >= image.shape[1]): right_edge = image.shape[1]-1
        bottom_edge = box.y1+box.height-1
        if (bottom_edge>= image.shape[0]): bottom_edge = image.shape[0]-1
        rr,cc = polygon_perimeter([box.y1, box.y1, bottom_edge, bottom_edge],
                                 [box.x1, right_edge, right_edge, box.x1])
        image[rr, cc] = color_rgb


def get_matches(target_box_list, potential_match_box_list):
    """
    Given a target list of FaceBox objects, finds the best match for each target
    from the potential_match_box_list

    Args:
        target_box_list: a list of FaceBox objects 
        potential_match_box_list: a list of FaceBox objects
    Returns:
        best_match: a list of FaceBoxMatch objects, where match is found by largest IoU
    """
    null_box = FaceBox()
    best_matches = [FaceBoxMatch(target_box, null_box) for target_box in target_box_list]
    for box_pair in best_matches:
        for potential_match_box in potential_match_box_list:
            old_iou = box_pair.target_box.iou(box_pair.match_box)
            new_iou = box_pair.target_box.iou(potential_match_box)
            if (new_iou > old_iou):
                box_pair.match_box = potential_match_box
    return best_matches
