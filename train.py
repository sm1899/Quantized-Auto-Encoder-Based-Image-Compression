import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.models import vgg19
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.models as resnet_models
import os



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
qb = 8  # Quantization bits


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for foldername in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, foldername)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.JPG') or filename.endswith('.PNG') or filename.endswith('.JPEG'):
                        img_path = os.path.join(folder_path, filename)
                        image = Image.open(img_path)
                        # Check if image has 3 channels
                        if image.mode == 'RGB':
                            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
    
# Define transformations
transform = transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset for training
train_dataset = CustomDataset(root_dir='/csehome/m23mac008/cvproject/imagenet-mini/train', transform=transform)

# Split training dataset into training and validation sets
train_size = 0.8  # 80% for training, 20% for validation
train_indices, val_indices = train_test_split(range(len(train_dataset)), train_size=train_size, random_state=42)

# Create dataset for validation
val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

# Create dataset for testing
test_dataset = CustomDataset(root_dir='/csehome/m23mac008/cvproject/imagenet-mini/val', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # Shuffle validation data
test_loader = DataLoader(test_dataset, batch_size=batch_size)



def vgg_loss(vgg: nn.Module,x, x_hat) :
    x_latent = vgg(x)
    x_hat_latent = vgg(x_hat)
    return torch.mean(torch.abs(x_latent - x_hat_latent))


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



# Create a random input tensor
batch_size = 4
channels = 3
height = 256
width = 256
random_input = torch.rand(batch_size, channels, height, width)

# Test the Encoder forward pass
encoder = Encoder(resnet_model_name="resnet101")
encoded = encoder(random_input)
print(f"Encoded shape: {encoded.shape}")

# Test the Decoder forward pass
decoder = Decoder(resnet_model_name="resnet101")
decoded = decoder(encoded)
print(f"Decoded shape: {decoded.shape}")


# Test the AutoEncoder forward pass
autoencoder = AutoEncoder(resnet_model_name="resnet101", qb=8)
output = autoencoder(random_input)
print(f"AutoEncoder output shape: {output.shape}")

# Define model, loss function, and optimizer
model = AutoEncoder(resnet_model_name="resnet101", qb=qb).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# VGG loss
# VGG loss
vgg = nn.Sequential(*list(vgg19(pretrained=True).features)[:36]).eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

# Training loop
train_losses = []
val_losses = []
psnr_values = []
ssim_values = []
bpp_values = []
mu = .1
for epoch in range(num_epochs):
    running_loss = 0.0

    # Training loop
    model.train()
    for images in train_loader:
        images = images.to(device)  # Move images to GPU if available

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)  # Reconstruction loss
        perceptual_loss = vgg_loss(vgg, images, outputs)  # Perceptual loss
        total_loss = loss + mu * perceptual_loss  # Combine losses

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

    # Save weights every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'weights3/autoencoder_{epoch + 1}.pth')
        

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    psnr_value = 0.0
    ssim_value = 0.0
    bpp_value = 0.0
    with torch.no_grad():
        for val_images in val_loader:
            val_images = val_images.to(device)
            val_outputs = model(val_images)
            loss = criterion(val_outputs, val_images)  # Reconstruction loss
            perceptual_loss = vgg_loss(vgg, val_images, val_outputs)  # Perceptual loss
            total_loss = loss + mu * perceptual_loss  # Combine losses
            val_loss += total_loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Validation Loss: {val_loss:.4f}')

# Save losses to a text file
with open('losses.txt', 'w') as f:
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        f.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')

print('Training completed')
