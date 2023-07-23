import re
import os
import sys
import copy
from skimage import io
from PIL import Image
import numpy as np
from AttackFace import tools

class ImageLabels:
    def __init__(self, img_path, true_box_list=[], found_box_dict={}, **kwargs):
        self.img_path = img_path
        if (not os.path.isfile(self.img_path)): 
            print('Attempting to create ImageLabels object for bad image path: ' 
                    + str(self.img_path))
            raise IOError
        try:
            image = Image.open(img_path)
            image = image.convert("RGB")
            image.verify() # verify that it is, in fact an image
        except:
            print('Input Image Read Error (ImageLabels init): ' + img_path)  
            raise
        image = np.array(image)
        self.img_shape = image.shape
        self.true_box_list = true_box_list
        self.found_box_dict = found_box_dict
        self.__dict__.update(kwargs)
        self.drawn_images = [] 
    
    def __str__(self):
        main_str = '(Img Path: \n{} \nTruth Box List: \n{} \nFound Box Dict: \n{}\n'.format(
            self.img_path, 
            self.true_box_list, 
            self.found_box_dict)
        kwargs_str = 'Attack Information:'
        for key, item in self.kwargs.items(): kwargs_str += '\n{}: {}'.format(key, item)
        kwargs_str += ')'
        return main_str + kwargs_str
    
    def __repr__(self):
        return str(self)
    
    def add_detector_labels(self,detector_dict):
        if (not os.path.isfile(self.img_path)): 
            sys.exit('Attempted to add detector labels to bad image: ' + self.img_path)
        found_box_dict = {}
        try:
            image = Image.open(self.img_path).convert("RGB")
            image.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Input Image Read Error (add_detector_labels): ' + img_path)
        image = np.array(image)
        for detector_name, detector in detector_dict.items():
            found_box_list = tools.get_found_boxes(image,detector)
            found_box_dict.update( {detector_name : found_box_list} )
        self.found_box_dict = found_box_dict
        return self
      

