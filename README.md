# Quantized Auto-Encoder-Based Image Compression

This repository contains the implementation of a quantized auto-encoder for image compression using PyTorch, along with scripts for training, testing, and evaluating the compression performance.

## Requirements

- Python 3.6 or higher
- PyTorch
- Torchvision
- scikit-learn
- scikit-image
- Pillow
- tqdm
- NumPy
- OpenCV
- arithmetic-compressor

Install the required packages using:
```
pip install -r requirements.txt
```
## Usage

### 1. Training

1. Prepare your dataset by organizing the images into separate folders for training and validation.
2. Update the paths in the `train.py` script to point to your dataset directories.
3. Adjust the hyperparameters (e.g., batch size, learning rate, quantization bits) as needed.
4. Run the training script:
  ```
  python train.py
  ```
5. weights will be saved in the weights folder in intervals of 5 epochs

The script will train the auto-encoder model, saving the weights every 5 epochs in the `weights3/` directory. The training and validation losses will be saved to `losses.txt`.

### 2. Testing

1. Place the images you want to compress in the `images` folder.
2. Run the `test.py` script:
   ```
   python test.py

   ```
This script will:

- Load the pre-trained auto-encoder model
- Compress each image in the `images` folder using the auto-encoder and arithmetic coding
- Save the compressed bitstreams in the `compressed_folder` directory
- Decompress the bitstreams and save the reconstructed images in the `decompressed_folder` directory

### 3. Evaluation

1. Place your original images in the `images` folder.
2. Place the output images in the `decompressed_folder` folder.
3. place the train weights in the weights folder
4. Run the `metrics.py` script:
   ```
   python metrics.py

   ```
This script will:

- Loop through all image files in the `images` and `decompressed_folder` folders
- Calculate the Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) values for each image pair (original and compressed)
- Print the SSIM and PSNR values for each image

## Model

The auto-encoder model used in this implementation consists of:

- Encoder: A pre-trained ResNet-101 encoder
- Decoder: A custom ResNet decoder
- Quantization: The latent space is quantized to reduce the bitrate

The model was trained using a combination of mean squared error and VGG perceptual loss, and the model outputs feature maps with 512 channels and 8 x 8 spatial dimensions. Then the feature maps are flattened and become a vector of size 32768. The vector is then quantized into `B` quantization levels.

### Train quantization

In the training phase, `noise` is appended to the input image. The `noise` is sampled from N(-0.5, 0.5) and then scaled by `B` quantization levels. So the final noise vector is:
```
scale = 2 ** -B
noise = (torch.randn(n) * 0.5 - 0.5) * scale
```
### Inference quantization

In the inference mode, the vector is quantized using `torch.clamp(0, 1)` and then scaled by `B` quantization levels. So the final quantized vector is:
```
quantized = torch.clamp(vector, 0, 1) * 2 ** B + 0.5
quantized = quantized.int()
```

## Credits

This implementation is based on the following papers:

- Alexandre, D., Chang, C.-P., Peng, W.-H., & Hang, H.-M. (2019). An Autoencoder-based Learned Image Compressor: Description of Challenge Proposal by NCTU. arXiv preprint arXiv:1902.07385.
- Wang, B., & Lo, K.-T. (2024). Autoencoder-based joint image compression and encryption. Journal of Information Security and Applications, 80, 103680. https://doi.org/10.1016/j.jisa.2023.103680

## Authors
Sougata Moi (M23MAC008)
Mitesh Kumar (M23MAC004)
Niraj Singha (M23MAC005)
Ratnesh Kumar Tiwari (M23MAC011)
## License

This project is licensed under the [MIT License](LICENSE).
