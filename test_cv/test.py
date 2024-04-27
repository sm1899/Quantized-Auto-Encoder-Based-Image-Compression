import torchvision.models as resnet_models
import torch.nn as nn
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision
import os
import typing as tp
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import pickle

qb = 8

class Encoder(nn.Module):
    def __init__(self, resnet_model_name: str):
        super().__init__()
        resnet = resnet_models.__dict__[resnet_model_name](pretrained=True)

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        return x

class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.05)
        
        if stride != 1:
            self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.norm2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.LeakyReLU(negative_slope=0.05)
            self.residual = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.LeakyReLU(negative_slope=0.05)
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.relu2(self.norm2(self.conv2(out)))
        res = self.residual(x)
        return out + res

class Decoder(nn.Module):
    def __init__(self, resnet_model_name: str):
        super().__init__()
        self.inchannel = 2048

        self.up = nn.Upsample(scale_factor=2)
        
        self.layer1 = nn.Sequential(
            ResidualUpsampleBlock(self.inchannel, 256, stride=2),
            ResidualUpsampleBlock(256, 256, stride=1)
        )
        
        self.layer2 = nn.Sequential(
            ResidualUpsampleBlock(256, 128, stride=2),
            ResidualUpsampleBlock(128, 128, stride=1)
        )
        
        self.layer3 = nn.Sequential(
            ResidualUpsampleBlock(128, 64, stride=2),
            ResidualUpsampleBlock(64, 64, stride=1)
        )
        
        self.layer4 = nn.Sequential(
            ResidualUpsampleBlock(64, 64, stride=1),
            ResidualUpsampleBlock(64, 64, stride=1)
        )

        self.resize = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.resize(out)
        out = self.sigmoid(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, resnet_model_name: str, qb: int):
        super().__init__()
        self.qb = qb
        self.encoder = Encoder(resnet_model_name=resnet_model_name)
        self.decoder = Decoder(resnet_model_name=resnet_model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        x = torch.clamp(x, 0.0, 1.0)
        x = x + (1 / 2 ** self.qb) * (torch.rand_like(x) * 0.5 - 0.5)

        x = self.decoder(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the pre-trained model
model = AutoEncoder(resnet_model_name="resnet101", qb=8)
model.load_state_dict(torch.load('weights_lr_e4_b_8/autoencoder_50_mu1', map_location=device))
model.eval().to(device)
Encoder = model.encoder
Decoder = model.decoder



def list_to_int(l: tp.List[int]) -> int:
    return int("1" + "".join(map(str, l)), 2)


def shape_to_str(shape: tp.List[int]) -> str:
    return "_".join(map(str, shape))


def save_compressed(compressed: tp.List[int], shape: tp.List[int], file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(f"{shape_to_str(shape)},{list_to_int(compressed)}", f)


def load_compressed(file_name) -> tp.Tuple[tp.List[int], tp.List[int]]:
    with open(file_name, "rb") as f:
        result = pickle.load(f)
    result = result.split(",")
    shape = list(map(int, result[0].split("_")))
    compressed = list(map(int, bin(int(result[1]))[3:]))
    return compressed, shape


def compress(vector: torch.Tensor, qb: int):
    """
    Compresses a vector of floats into a bitstring.
    :param vector: The vector to compress.
    :param qb: The number of quantization levels to use.
    :return: The compressed bitstring.
    """
    shape = vector.shape
    vector = vector.flatten()
    vector = torch.clamp(vector, 0.0, 1.0)
    vector = (vector * (2 ** qb) + 0.5).to(torch.int64)
    vector = vector.tolist()

    keys = [key for key in range(0, 2 ** qb + 1)]
    prob = 1.0 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)

    return coder.compress(vector), list(shape)


def decompress(compressed: tp.List[int], shape: tp.List[int], qb: int) -> torch.Tensor:
    length = np.prod(shape)
    keys = [key for key in range(0, 2 ** qb + 1)]
    prob = 1.0 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)
    decompressed = coder.decompress(compressed, length)
    decompressed = np.fromiter(map(int, decompressed), dtype=np.int64)
    decompressed = torch.from_numpy(decompressed).float()
    decompressed = decompressed / (2 ** qb)
    decompressed = decompressed.view(1, *shape)

    return decompressed



import sys

# Increase the limit to a larger number
sys.set_int_max_str_digits(1000000)  # or any number that suits your needs

input_folder = 'images'
compressed_folder = 'compressed_folder'
decompressed_folder = 'decompressed_folder'

# Create the output folders if they don't exist
os.makedirs(compressed_folder, exist_ok=True)
os.makedirs(decompressed_folder, exist_ok=True)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image file
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image).unsqueeze(0).to(device)

        # Compression
        with torch.no_grad():
            image = Encoder(image)

        image = image.squeeze(0).cpu()
        compressed, shape = compress(image, qb)
        compressed_filename = os.path.splitext(filename)[0] + '.bin'
        compressed_path = os.path.join(compressed_folder, compressed_filename)
        save_compressed(compressed, shape, compressed_path)

        # Decompression
        vector, shape = load_compressed(compressed_path)
        vector = decompress(vector, shape, qb)

        with torch.no_grad():
            vector = vector.to(device)
            image = Decoder(vector)

        image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        image = image * 255.0
        image = image.astype(np.uint8)

        image = Image.fromarray(image)
        decompressed_path = os.path.join(decompressed_folder, filename)
        image.save(decompressed_path)

        