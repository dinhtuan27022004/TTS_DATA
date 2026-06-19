import os
import argparse
import sys
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Add parent directory of src to sys.path to allow execution from project root or src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TimePreservingMelAutoencoder
from src.utils import load_config, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Reconstruct Mel Spectrogram")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--wav_path", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--out_dir", type=str, default="output_reconstruct", help="Directory to save output files")
    parser.add_argument("--plot", action="store_true", help="Plot original and reconstructed mel spectrograms")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        return
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = TimePreservingMelAutoencoder(
        n_mels=config["n_mels"],
        latent_dim=config["latent_dim"]
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    epoch, loss = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded model from epoch {epoch} with loss {loss:.4f}")
    
    model.eval()
    
    # Load and preprocess WAV
    if not os.path.exists(args.wav_path):
        print(f"Error: WAV file not found at '{args.wav_path}'")
        return
        
    waveform, sr = torchaudio.load(args.wav_path)
    print(f"Loaded WAV with shape: {waveform.shape}, Sample Rate: {sr}Hz")
    
    # Convert multi-channel to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print("Converted multi-channel audio to mono.")
        
    # Resample if needed
    target_sr = config["sample_rate"]
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        print(f"Resampled audio from {sr}Hz to {target_sr}Hz.")
        
    # Extract Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=config["n_fft"],
        win_length=config["win_length"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        power=1.0
    ).to(device)
    
    waveform = waveform.to(device)
    with torch.no_grad():
        mel = mel_transform(waveform)  # Shape: [1, n_mels, T]
        log_mel = torch.log(torch.clamp(mel, min=1e-5))  # Shape: [1, n_mels, T]
        
        # Forward pass (Encode -> Decode)
        mel_recon, z = model(log_mel)
        
    # Convert outputs to NumPy arrays
    log_mel_np = log_mel.squeeze(0).cpu().numpy()  # [n_mels, T]
    mel_recon_np = mel_recon.squeeze(0).cpu().numpy()  # [n_mels, T]
    z_np = z.squeeze(0).cpu().numpy()  # [latent_dim, T]
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save files as .npy
    orig_path = os.path.join(args.out_dir, "original_mel.npy")
    recon_path = os.path.join(args.out_dir, "reconstructed_mel.npy")
    latent_path = os.path.join(args.out_dir, "latent_z.npy")
    
    np.save(orig_path, log_mel_np)
    np.save(recon_path, mel_recon_np)
    np.save(latent_path, z_np)
    
    print(f"Saved original mel spectrogram to: {orig_path}")
    print(f"Saved reconstructed mel spectrogram to: {recon_path}")
    print(f"Saved latent representation z to: {latent_path}")

    # Reconstruct audio using Vocos neural vocoder (like F5-TTS)
    print("🔊 Reconstructing audio via Vocos...")
    try:
        from vocos import Vocos
        
        # Load pretrained Vocos on correct device
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        vocos.eval()
        
        # vocos expects log-mel spectrogram tensor shape: [B, 100, T]
        # If the model has 80 channels, we dynamically interpolate to 100 channels for Vocos
        if log_mel.shape[1] == 80:
            print("⚠️ Warning: Input mel spectrogram has 80 channels (from old checkpoint/config). Dynamically interpolating to 100 channels for Vocos...")
            log_mel_for_vocos = torch.nn.functional.interpolate(
                log_mel.unsqueeze(1), size=(100, log_mel.shape[2]), mode='bilinear', align_corners=True
            ).squeeze(1)
            mel_recon_for_vocos = torch.nn.functional.interpolate(
                mel_recon.unsqueeze(1), size=(100, mel_recon.shape[2]), mode='bilinear', align_corners=True
            ).squeeze(1)
        else:
            log_mel_for_vocos = log_mel
            mel_recon_for_vocos = mel_recon

        with torch.no_grad():
            orig_wav_tensor = vocos.decode(log_mel_for_vocos.to(device).to(torch.float32))
            recon_wav_tensor = vocos.decode(mel_recon_for_vocos.to(device).to(torch.float32))
            
        orig_wav_path = os.path.join(args.out_dir, "original_vocos.wav")
        recon_wav_path = os.path.join(args.out_dir, "reconstructed_vocos.wav")
        
        torchaudio.save(orig_wav_path, orig_wav_tensor.cpu(), 24000)
        torchaudio.save(recon_wav_path, recon_wav_tensor.cpu(), 24000)
        print(f"Saved original baseline audio (Vocos) to: {orig_wav_path}")
        print(f"Saved reconstructed audio (Vocos) to: {recon_wav_path}")
    except Exception as e:
        print(f"Warning: Failed to reconstruct audio via Vocos: {e}")
    
    # Optionally plot and save visual comparisons
    if args.plot:
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.imshow(log_mel_np, aspect='auto', origin='lower', cmap='viridis')
        plt.title("Original Log-Mel Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 1, 2)
        plt.imshow(mel_recon_np, aspect='auto', origin='lower', cmap='viridis')
        plt.title("Reconstructed Log-Mel Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        
        # Absolute difference (error map)
        diff_np = np.abs(log_mel_np - mel_recon_np)
        mean_l1_error = np.mean(diff_np)
        
        plt.subplot(3, 1, 3)
        plt.imshow(diff_np, aspect='auto', origin='lower', cmap='inferno')
        plt.title(f"Absolute Reconstruction Error (Mean L1 Error: {mean_l1_error:.4f})")
        plt.colorbar()
        
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, "reconstruction_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved visual comparison plot to: {plot_path}")

if __name__ == "__main__":
    main()
