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


def to_std_float(image):
    """Converts image to 0 to 1 float to avoid wrapping that occurs with uint8"""
    image.astype(np.float16, copy=False)
    image = np.multiply(image, 1/255)
    return image

def to_std_uint8(image):
    """Properly handles the conversion to uint8"""
    image = cv2.convertScaleAbs(image, alpha=255)
    return image

def display_image(image, title='Image'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title(title)
    plt.show()

def add_salt_and_pepper(image, noise_prob=0.1, show=True):
    """Converts pixels of `image` to black or white independently each with probability `noise_prob`"""
    image = to_std_float(image)
    white_value = int(2 / noise_prob)
    noise = np.random.randint(white_value + 1, size=(image.shape[0], image.shape[1], 1))
    image = np.where(noise == 0, 0, image)
    image = np.where(noise == white_value, 1, image)
    image = to_std_uint8(image)
    if show:
        display_image(image, 'Image with Salt & Pepper Noise')
    return image

def add_gaussian_noise(image, std_dev=0.15, show=True):
    image = to_std_float(image)
    noise = np.random.normal(0, std_dev, (image.shape[0],image.shape[1], 3))
    image += noise
    image = to_std_uint8(image)
    if show:
        display_image(image, 'Image with Gaussian Noise')
    return image

def show_median_blur(image, kernel_size=3, title ='Median Blur Result'):
    image = cv2.medianBlur(image, kernel_size)
    display_image(image, title)
    return image

def show_mean_blur(image, kernel_size=(5, 5), title ='Mean Filter Result'):
    image = cv2.blur(image, kernel_size)
    display_image(image, title)
    return image

def show_gaussian_blur(image, kernel_size=(5, 5), std_dev=0, title ='Gaussian Smoothing Result'):
    image = cv2.GaussianBlur(image, kernel_size, std_dev)
    display_image(image, title)
    return image