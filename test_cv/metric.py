import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Set the paths to the two folders
original_folder = 'images'
compressed_folder = 'decompressed_folder_mu1_e50'

# Loop through the files in the original folder
for filename in os.listdir(original_folder):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the original and compressed images
        original_image = cv2.imread(os.path.join(original_folder, filename))
        compressed_image = cv2.imread(os.path.join(compressed_folder, filename))

        # Calculate SSIM
        ssim_value = ssim(original_image, compressed_image, multichannel=True,channel_axis=2)
        print(f'SSIM for {filename}: {ssim_value}')

        # Calculate PSNR
        mse = np.mean((original_image - compressed_image) ** 2)
        if mse == 0:
            psnr_value = float('inf')
        else:
            psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
        print(f'PSNR for {filename}: {psnr_value}')

  