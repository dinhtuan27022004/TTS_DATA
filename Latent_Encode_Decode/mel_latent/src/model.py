import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    """
    Residual block utilizing 1D Convolutions, Group Normalization, and SiLU activations.
    Ensures that the time dimension T remains unchanged.
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # Padding calculation to preserve the sequence length T
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        # Using GroupNorm which splits channels into groups (ideal for variable length batch processing in 1D)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act2(x + res)

class Encoder(nn.Module):
    """
    Encoder that maps a mel-spectrogram [B, 80, T] to a latent space [B, 64, T].
    Uses 1D Convolutions and Residual Blocks with dilations to capture context.
    """
    def __init__(self, in_channels=80, hidden_channels=256, latent_channels=64):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=hidden_channels)
        self.act = nn.SiLU()
        
        # Dilated residual blocks to build a wider receptive field without pooling/striding
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_channels, kernel_size=3, dilation=1),
            ResBlock1D(hidden_channels, kernel_size=3, dilation=3),
            ResBlock1D(hidden_channels, kernel_size=3, dilation=5),
            ResBlock1D(hidden_channels, kernel_size=3, dilation=1),
        ])
        
        self.out_conv = nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Input shape: [B, in_channels, T]
        x = self.in_conv(x)
        x = self.norm(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)
        # Output shape: [B, latent_channels, T]
        return x

class Decoder(nn.Module):
    """
    Decoder that maps the latent space [B, 64, T] back to the mel-spectrogram space [B, 80, T].
    """
    def __init__(self, latent_channels=64, hidden_channels=256, out_channels=80):
        super().__init__()
        self.in_conv = nn.Conv1d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=hidden_channels)
        self.act = nn.SiLU()
        
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_channels, kernel_size=3, dilation=1),
            ResBlock1D(hidden_channels, kernel_size=3, dilation=3),
            ResBlock1D(hidden_channels, kernel_size=3, dilation=5),
            ResBlock1D(hidden_channels, kernel_size=3, dilation=1),
        ])
        
        self.out_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # Input shape: [B, latent_channels, T]
        x = self.in_conv(x)
        x = self.norm(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)
        # Output shape: [B, out_channels, T]
        return x

class TimePreservingMelAutoencoder(nn.Module):
    """
    Main Autoencoder Class.
    """
    def __init__(self, n_mels=80, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(in_channels=n_mels, hidden_channels=hidden_dim, latent_channels=latent_dim)
        self.decoder = Decoder(latent_channels=latent_dim, hidden_channels=hidden_dim, out_channels=n_mels)

    def encode(self, mel):
        """Encodes mel spectrogram [B, n_mels, T] to latent z [B, latent_dim, T]"""
        return self.encoder(mel)

    def decode(self, z):
        """Decodes latent z [B, latent_dim, T] to reconstructed mel [B, n_mels, T]"""
        return self.decoder(z)

    def forward(self, mel):
        """Performs full forward pass returning reconstructed mel and latent z."""
        z = self.encode(mel)
        mel_recon = self.decode(z)
        return mel_recon, z
