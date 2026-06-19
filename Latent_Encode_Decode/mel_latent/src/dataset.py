import os
import glob
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, wav_dir, sample_rate=24000, n_mels=80, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Search for all wav files in wav_dir (including subdirectories)
        self.wav_paths = glob.glob(os.path.join(wav_dir, "**/*.wav"), recursive=True)
        if len(self.wav_paths) == 0:
            # Fallback for flat directory structures
            self.wav_paths = glob.glob(os.path.join(wav_dir, "*.wav"))
            
        # Define the MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=1.0  # Use magnitude spectrogram (standard for F5-TTS / Librosa log-mel)
        )
        
    def __len__(self):
        return len(self.wav_paths)
        
    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        waveform, sr = torchaudio.load(wav_path)
        
        # Convert multi-channel (stereo) audio to mono by taking the mean across channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample waveform if its sample rate doesn't match the target sample rate
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        # Extract Mel Spectrogram
        # waveform shape: [1, T_samples] -> mel shape: [1, n_mels, T_frames]
        mel = self.mel_transform(waveform)
        mel = mel.squeeze(0)  # Shape: [n_mels, T_frames]
        
        # Log scale: ln(mel + eps). Standard silence value is log(1e-5) ~= -11.5129
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return log_mel

def collate_fn(batch):
    """
    Collate function to dynamic pad the batch items along the time dimension T.
    Pads with log(1e-5) = -11.5129 which represents silence.
    """
    # batch is a list of log-mel tensors: [[80, T_1], [80, T_2], ...]
    max_t = max(mel.shape[1] for mel in batch)
    
    padded_batch = []
    for mel in batch:
        pad_size = max_t - mel.shape[1]
        if pad_size > 0:
            # Pad on the right side of the time dimension (dimension 1 of shape [80, T])
            # F.pad expects padding for the last dimension to be (left_pad, right_pad)
            mel_padded = F.pad(mel, (0, pad_size), mode='constant', value=-11.512925)
        else:
            mel_padded = mel
        padded_batch.append(mel_padded)
        
    # Stack into a batch: [B, 80, T_max]
    return torch.stack(padded_batch, dim=0)
