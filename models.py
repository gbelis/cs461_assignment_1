import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

from PIL import Image

class ImageEncoder(nn.Module):
    """
    A simple neural network template for self-supervised learning.

    Structure:
    1. Encoder: Maps an input image of shape 
       (input_channels, input_dim, input_dim) 
       into a lower-dimensional feature representation.
    2. Projector: Transforms the encoder output into the final 
       embedding space of size `proj_dim`.

    Notes:
    - DO NOT modify the fixed class variables: 
      `input_dim`, `input_channels`, and `feature_dim`.
    - You may freely modify the architecture of the encoder 
      and projector (layers, activations, normalization, etc.).
    - You may add additional helper functions or class variables if needed.
    """

    ####### DO NOT MODIFY THE CLASS VARIABLES #######
    input_dim: int = 64
    input_channels: int = 3
    feature_dim: int = 512#1000
    proj_dim: int = 128
    #################################################

    def __init__(self):
        super().__init__()
        
        hidden_dim = 512
        enc = resnet18(weights=None)
        enc.conv1 = nn.Conv2d(self.input_channels, 64, 3, 1, 1, bias=False)
        enc.maxpool = nn.Identity()
        enc.fc = nn.Linear(hidden_dim, self.feature_dim)
        self.encoder = enc
        

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.proj_dim, bias=False),
            nn.BatchNorm1d(self.proj_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.proj_dim)
        )

    def forward(self, x):
        """
        x: (batch_size, channels, height, width) tensor of images
        returns: repr (batch_size, feature_dim), proj (batch_size, proj_dim), pred (batch_size, proj_dim)
        """
        # TODO
        feature = self.encoder(x)
        proj = self.projector(feature)
        pred = self.predictor(proj)

        proj = F.normalize(proj, dim=-1, eps=1e-8)
        pred = F.normalize(pred, dim=-1, eps=1e-8)

        return feature, proj, pred
    
    def get_features(self, x):
        """
        Get the features from the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output features of shape (batch_size, feature_dim).
        """
        features = self.encoder(x)
        return features