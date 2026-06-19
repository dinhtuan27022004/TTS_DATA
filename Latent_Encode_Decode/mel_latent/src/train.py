import os
import argparse
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory of src to sys.path to allow execution from project root or src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import MelDataset, collate_fn
from src.model import TimePreservingMelAutoencoder
from src.utils import load_config, save_checkpoint, load_checkpoint, count_parameters, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mel Latent Autoencoder")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable default resume from checkpoint_dir/model.pt",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        return
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare paths
    # Resolve relative paths relative to config file's directory if necessary, 
    # but here we assume paths are relative to the root directory from which the script is run.
    wav_dir = config["wav_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir, exist_ok=True)
        print(f"Warning: Created empty WAV directory at '{wav_dir}'. Please put some .wav files in it.")
        
    # Initialize dataset
    dataset = MelDataset(
        wav_dir=wav_dir,
        sample_rate=config["sample_rate"],
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"]
    )
    
    if len(dataset) == 0:
        print(f"Error: No WAV files found in the directory '{wav_dir}'. Please add .wav files and try again.")
        return
        
    print(f"Found {len(dataset)} WAV files for training.")
    
    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = TimePreservingMelAutoencoder(
        n_mels=config["n_mels"],
        latent_dim=config["latent_dim"]
    ).to(device)
    
    print(f"Model created. Total trainable parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    
    # Resume from explicit checkpoint, otherwise default to checkpoint_dir/model.pt if it exists.
    start_epoch = 1
    latest_checkpoint = os.path.join(checkpoint_dir, "model.pt")
    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume and os.path.isfile(latest_checkpoint):
        resume_path = latest_checkpoint

    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}")
        start_epoch, _ = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")
        
    num_epochs = config["num_epochs"]
    save_every = config["save_every"]
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_mse = 0.0
        epoch_latent = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for step, batch_mel in enumerate(progress_bar):
            # batch_mel shape: [B, 80, T]
            batch_mel = batch_mel.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mel_recon, z = model(batch_mel)
            
            # Loss computation
            l1_loss = torch.mean(torch.abs(mel_recon - batch_mel))
            mse_loss = torch.mean((mel_recon - batch_mel) ** 2)
            # Latent regularization to keep z values bounded and stable
            latent_loss = torch.mean(z ** 2) * 1e-4
            
            loss = l1_loss + 0.5 * mse_loss + latent_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_l1 += l1_loss.item()
            epoch_mse += mse_loss.item()
            epoch_latent += latent_loss.item()
            
            # Update progress bar description
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "l1": f"{l1_loss.item():.4f}",
                "mse": f"{mse_loss.item():.4f}",
                "latent": f"{latent_loss.item():.4f}"
            })
            
        # Log epoch summary
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_l1 = epoch_l1 / num_batches
        avg_mse = epoch_mse / num_batches
        avg_latent = epoch_latent / num_batches
        
        print(f"Epoch {epoch}/{num_epochs} Summary - Avg Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | MSE: {avg_mse:.4f} | Latent: {avg_latent:.4f}")
        
        # Save checkpoints
        if epoch % save_every == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            save_checkpoint(model, optimizer, epoch, avg_loss, latest_checkpoint)
            print(f"Updated latest checkpoint at {latest_checkpoint}")

if __name__ == "__main__":
    main()
