# Time-Preserving Mel Latent Autoencoder

A PyTorch project to train a Time-Preserving Mel Autoencoder that learns a channel-bottlenecked latent representation $z \in \mathbb{R}^{B \times 128 \times T}$ from log-mel spectrograms $x \in \mathbb{R}^{B \times 80 \times T}$.

## 📂 Project Structure
```
mel_latent/
├── configs/
│   └── config.yaml          # All hyperparameters, paths, and training configurations
├── data/
│   └── wavs/                # Put your training .wav files here
├── checkpoints/             # Directory where PyTorch model checkpoints are saved
├── src/
│   ├── dataset.py           # Audio loading, resampling, log-mel extraction, and dynamic collate/padding
│   ├── model.py             # TimePreservingMelAutoencoder, Encoder, Decoder, ResBlock1D definitions
│   ├── train.py             # Training loop, compound loss, and checkpoint saving/resuming
│   ├── utils.py             # Configuration loader, checkpoint save/load, seed, and parameter counting
│   └── inference_reconstruct.py # Reconstruct log-mel from wav, save arrays, and plot comparison
└── requirements.txt         # Required python packages
```

## 🛠️ Step-by-Step Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Put your `.wav` files into the training directory:
```bash
mkdir -p data/wavs
```

### 3. Run Training
```bash
# Start training from scratch
python3 src/train.py --config configs/config.yaml

# Resume training from a checkpoint
python3 src/train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_20.pt
```

### 4. Run Reconstruction Inference
```bash
python3 src/inference_reconstruct.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/checkpoint_epoch_100.pt \
    --wav_path path/to/your/test_audio.wav \
    --out_dir output_reconstruct \
    --plot
```

For more details on the architecture, latent space interpretation, evaluation metrics, and integration with F5-TTS, see the generated walkthrough guide.
