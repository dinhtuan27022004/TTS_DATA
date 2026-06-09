import os
import subprocess
import sys

BASE_DIR = "/home/reg/TTS_DATA"
MOCK_VOCAB_DIR = os.path.join(BASE_DIR, "Fish_speech/checkpoints/openaudio-s1-mini")
MOCK_VOCAB_PATH = os.path.join(MOCK_VOCAB_DIR, "vocab.json")
MOCK_DATA_DIR = os.path.join(BASE_DIR, "Processed_DATA_mock")
CONFIG_PATH = os.path.join(BASE_DIR, "Fish_speech/configs/finetune_vn_lora.yaml")
MOCK_CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs/vn_finetuned/checkpoint.pt")

def run_cmd(cmd, description):
    print(f"\n==========================================")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"==========================================")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("Stdout:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print("Stdout:")
        print(e.stdout)
        print("Stderr:")
        print(e.stderr)
        return False

def main():
    # 1. Update vocab
    vocab_cmd = [sys.executable, "Fish_speech/update_vocab.py", "--vocab-path", MOCK_VOCAB_PATH]
    if not run_cmd(vocab_cmd, "Vocabulary Update / Dialect Tag Injection"):
        sys.exit(1)
        
    # Check if vocab was updated properly
    with open(MOCK_VOCAB_PATH, "r") as f:
        import json
        vocab = json.load(f)
    print(f"Verified vocab tags:")
    for tag in ["<bắc>", "<trung>", "<nam>"]:
        print(f"  {tag}: {vocab.get(tag)}")
        
    # 2. Extract VQ tokens
    extract_cmd = [sys.executable, "Fish_speech/tools/vqgan/extract_vq.py", MOCK_DATA_DIR, "--checkpoint-path", "Fish_speech/checkpoints/openaudio-s1-mini/codec.pth"]
    if not run_cmd(extract_cmd, "VQGAN Semantic Token Extraction (Mock)"):
        sys.exit(1)
        
    # Verify token files exist
    token_files = glob_tokens()
    print(f"Generated token files: {len(token_files)}")
    for tf in token_files[:4]:
        print(f"  {os.path.basename(tf)}")
        
    # 3. Train model
    train_cmd = [
        sys.executable, "Fish_speech/train.py",
        "--config", CONFIG_PATH,
        "--train-data", os.path.join(MOCK_DATA_DIR, "vq_tokens"),
        "--val-data", os.path.join(MOCK_DATA_DIR, "vq_tokens")
    ]
    if not run_cmd(train_cmd, "LoRA Training Forward Pass (Mock)"):
        sys.exit(1)
        
    # 4. Inference
    infer_cmd = [
        sys.executable, "Fish_speech/inference.py",
        "--model", MOCK_CHECKPOINT_PATH,
        "--input", "<nam> Tôi đang ở miền Nam.",
        "--output", "output_synth.wav"
    ]
    if not run_cmd(infer_cmd, "Inference Synthesis (Mock)"):
        sys.exit(1)
        
    # Verify wav file was created
    if os.path.exists("output_synth.wav"):
        print("\nSuccess! output_synth.wav was successfully generated.")
        print(f"File size: {os.path.getsize('output_synth.wav')} bytes")
    else:
        print("\nError: output_synth.wav was not found.")
        sys.exit(1)

def glob_tokens():
    import glob
    return glob.glob(os.path.join(MOCK_DATA_DIR, "vq_tokens", "**", "*.npy"), recursive=True)

if __name__ == "__main__":
    main()
