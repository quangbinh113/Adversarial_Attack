import argparse
import time
import os
import numpy as np
import cv2
import torch
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import TensorDataset

from model import *
from dataset import *
from loss import *

# Create the argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--n_blocks', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./output_file/')
parser.add_argument('--input_dir', type=str, default='./input_images/')
parser.add_argument('--model_dir', type=str, default='./pretrained_model/')
# Add more arguments if needed

args = parser.parse_args()

if __name__ == "__main__":
    # Load the test image
    input_image_path = args.input_dir
    test = Image.open(input_image_path)
    timg = np.array(test) / 255
    # Assuming you have imported or defined addGaussNoise and nonlinear_filter functions
    # timg = addGaussNoise(timg, 50)
    # timg = nonlinear_filter(timg)
    timg = torch.tensor(timg.transpose(2, 0, 1)).float().unsqueeze(0)

    # Load the pre-trained model and perform denoising
    with torch.no_grad():
        timg = timg.to(args.device)
        model_path = args.model_dir
        G = Generator(args.n_blocks)
        G.load_state_dict(torch.load(model_path, map_location=args.device))
        G.eval()

        denoised_img = G(timg)

    # Convert the denoised image tensor back to a NumPy array and save it
    denoised_img_np = (denoised_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'denoised.png')
    img_denoised = Image.fromarray(denoised_img_np)
    img_denoised.save(output_path)
