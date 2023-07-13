import re
import os
import sys
import copy
import numpy as np
from skimage import io,util
from PIL import Image
from AttackFace import tools, imagelabels


class ImgNoisePair():
    """
    A class object for storing attacked image and noise array pairs.
    Expect img and noise to be np.array() objects

    Attributes:
        img: an array description of the image with a noise attack applied
        noise: an array description of the noise attack 
    """
    def __init__(self,img,noise):
        self.img = img
        self.noise = noise
def create_noisy_image(original_img_path,  output_dir, use_mult_noise=False):
    """
    Given a path to an image, create a new image with normally-distributed noise randomly applied

    Args:
        original_img_path: the path of the image you want to attack
        epsilon: an integer giving the maximum variation (out of 255)
        output_dir: the directory where you'd like to store the attacked image
        use_mult_noise: a bool indicating whether to use multiplicative noise
    Returns:
        image_file_path: the path to the output image you just created
    """
    epsilon = np.random.choice([176,112,16,32,224,48,64,96])
    if (not os.path.isfile(original_img_path)):
        sys.exit('Bad input image path: ' + original_img_path)
    if (not os.path.exists(output_dir)): os.makedirs(output_dir)
    try:
        image = Image.open(original_img_path).convert("RGB")
        image.verify() # verify that it is, in fact an image
    except (IOError,SyntaxError) as e:
        raise ValueError('Input Image Read Error (create_noisy_image): '
            + original_img_path + ': ' + str(e))
    image = copy.deepcopy(np.array(image))
    pixels = np.asarray(image).astype(np.float32)
    pixels /= 255.0
    noise = (epsilon/255.0) * np.random.normal(loc=0.0, scale=1.0, size=pixels.shape)
    image_pixels = pixels * (1.0 + noise) if (use_mult_noise) else pixels + noise
    image_pixels = np.clip(image_pixels, 0.0, 1.0)
    image_pixels *= 255
    image_pixels = np.asarray(image_pixels).astype(np.uint8)
    noise = np.clip(noise, 0.0, 1.0)
    noise *= 255
    noise = np.asarray(noise).astype(np.uint8)
    file_num = re.findall(r'[0-9]+[0-9]+\.jpg', original_img_path)[0]
    image_file_name = file_num
    image_file_path = os.path.join(output_dir,image_file_name)

    im_pix = Image.fromarray(image_pixels)
    im_pix.save(image_file_path)

    return image_file_path
def create_noisy_face(facebox, original_img_path, epsilon, n_tests, output_dir, 
        use_fixed_random_seed=True,use_mult_noise=False):
    """
    Given an image path and FaceBox, create an array of attacked image and noise array pairs

    Args:
        facebox: a FaceBox where we want to apply noise
        original_img_path: the path of the image you want to attack
        epsilon: an integer giving the maximum variation (out of 255)
        n_tests: the number of attacks you'd like to test 
        output_dir: the directory where you'd like to store the attacked image
        use_mult_noise: a bool indicating whether to use multiplicative noise
    Returns:
        image_noise_pairs: a list of ImgNoisePair objects, pairing attacked image and noise arrays
    """
    if (not os.path.isfile(original_img_path)):
        sys.exit('Bad input image path: ' + original_img_path)
    if (not os.path.exists(output_dir)): os.makedirs(output_dir)
    if (facebox.width == 0 or facebox.height == 0):
        raise ValueError('Got a facebox with width or height 0 (size w' + str(facebox.width)
            + ' h' + str(facebox.height) + '): ' + original_img_path)
    try:
        image = Image.open(original_img_path).convert("RGB")
        image.verify() # verify that it is, in fact an image
    except (IOError,SyntaxError) as e:
        raise ValueError('Input Image Read Error (create_noisy_face): '
            + original_img_path + ': ' + str(e))
    image = copy.deepcopy(np.array(image))
    x1 = facebox.x1
    y1 = facebox.y1
    if (x1 < 0): x1 = 0
    if (y1 < 0): y1 = 0
    x2 = x1 + facebox.width
    y2 = y1 + facebox.height
    if (x2 >= image.shape[1]): x2 = image.shape[1] -1
    if (y2 >= image.shape[0]): y2 = image.shape[0] -1
    face_size = (y2-y1,x2-x1,3)
    pixels = np.asarray(image).astype(np.float32)
    pixels /= 255.0
    image_noise_pairs = []
    if (use_fixed_random_seed): np.random.seed(12345)
    for test_num in range(n_tests):
        noise = np.zeros(pixels.shape)
        noise[y1:y2, x1:x2] = (epsilon/255.0) * np.random.normal(loc=0.0, scale=1.0, size=face_size)
        image_pixels = pixels * (1.0 + noise) if (use_mult_noise) else pixels + noise
        image_pixels = np.clip(image_pixels, 0.0, 1.0)
        image_pixels *= 255
        image_pixels = np.asarray(image_pixels).astype(np.uint8)
        noise = np.clip(noise, 0.0, 1.0)
        noise *= 255
        noise = np.asarray(noise).astype(np.uint8)
        image_noise_pairs.append(ImgNoisePair(image_pixels,noise))
    return image_noise_pairs 
