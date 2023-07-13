import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_psnr_min = 0
        self.delta = delta
        self.checkpoint_perf = []

    def __call__(self, g, d, train_psnr, val_psnr):

        score = val_psnr
        self.early_stop = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(g, d, val_psnr)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
                self.best_score = None
                self.val_psnr_min = 0
        else:
            self.best_score = score
            self.save_checkpoint(g, d, val_psnr)
            self.counter = 0
            self.checkpoint_perf = [train_psnr, val_psnr]
        return self.checkpoint_perf

    def save_checkpoint(self, g, d, val_psnr):
        self.val_psnr_min = val_psnr
        if self.verbose:
            print(f'Validation PSNR increased ({self.val_psnr_min:.6f} --> {val_psnr:.6f}).  Saving model ...')
            torch.save(g.state_dict(), 'Generator.pth')
            torch.save(d.state_dict(), 'Discriminator.pth')
        else:
            torch.save(g.state_dict(), 'Generator.pth')
            torch.save(d.state_dict(), 'Discriminator.pth')


from typing import List, Tuple
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

COLORS = ('blue', 'green', 'red')

def to_std_float(image: np.ndarray) -> np.ndarray:
    """Converts image to 0 to 1 float to avoid wrapping that occurs with uint8"""
    image.astype(np.float16, copy=False)
    image = np.multiply(image, 1/255)
    return image

def to_std_uint8(image: np.ndarray) -> np.ndarray:
    """Properly handles the conversion to uint8"""
    image = cv2.convertScaleAbs(image, alpha=255)
    return image

def show_images(images: List[np.ndarray], labels: List[str]=None, max_num_cols: int=4, fig_size: Tuple[int, int]=None):
    """
    Shows a grid of images (color space = BGR, shape = (h, w, c))
    """
    ncols = min(len(images), max_num_cols)
    nrows = math.ceil(len(images) / ncols)
    if not fig_size:
        fig_size = (5 * ncols, 5 * nrows)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=fig_size)
    for i, image in enumerate(images):
        ax = axs[divmod(i, ncols)]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if labels:
            ax.set_title(labels[i])

def show_image_histograms(images: List[np.ndarray], labels: List[str]=None, max_freq: int=None, max_num_cols: int=4, fig_size: Tuple[int, int]=None):
    """
    Shows a grid of image histograms (color space = BGR, shape = (h, w, c))
    """
    ncols = min(len(images), max_num_cols)
    nrows = math.ceil(len(images) / ncols)
    if not fig_size:
        fig_size = (5 * ncols, 5 * nrows)
    if not max_freq:
        max_freq = images[0].size / 16
    fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=fig_size)
    for i, image in enumerate(images):
        ax = axs[divmod(i, ncols)]
        for c in range(3):
            ax.hist(image[:, :, c].flatten(), color=COLORS[c], bins=256, label=COLORS[c], histtype='step', alpha=0.5)
            ax.set_ylim(0, max_freq)
        if labels:
            ax.set_title(labels[i])

def add_salt_and_pepper(image: np.ndarray, noise_prob: float=0.1) -> np.ndarray:
    """Converts pixels of `image` to black or white independently each with probability `noise_prob`"""
    image = to_std_float(image)
    white_value = int(2 / noise_prob)
    noise = np.random.randint(white_value + 1, size=(image.shape[0], image.shape[1], 1))
    image = np.where(noise == 0, 0, image)
    image = np.where(noise == white_value, 1, image)
    image = to_std_uint8(image)
    return image

def add_gaussian_noise(image: np.ndarray, std_dev: float=0.2):
    image = to_std_float(image)
    noise = np.random.normal(0, std_dev, (image.shape[0],image.shape[1], 3))
    image += noise
    image = to_std_uint8(image)
    return image