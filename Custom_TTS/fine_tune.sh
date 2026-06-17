#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SOURCE_PAIRS_DIR="${SOURCE_PAIRS_DIR:-/home/reg/TTS_DATA/data/YTDT2}"
DATASET_NAME="${DATASET_NAME:-$(basename "$SOURCE_PAIRS_DIR")}"
CONFIG_NAME="${CONFIG_NAME:-SemanticF5TTS_Base.yaml}"

BASE_VOCAB_PATH="${BASE_VOCAB_PATH:-/home/reg/TTS_DATA/Custom_TTS/data/my_dataset/vocab.txt}"
FIXED_VOCAB_SIZE="${FIXED_VOCAB_SIZE:-}"

PREPARE_DATA="${PREPARE_DATA:-auto}"
PREPARE_SEMANTIC="${PREPARE_SEMANTIC:-1}"
SEMANTIC_MODEL="${SEMANTIC_MODEL:-microsoft/wavlm-base-plus}"
SEMANTIC_LAYER="${SEMANTIC_LAYER:--1}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-4}"
SEMANTIC_DEVICE="${SEMANTIC_DEVICE:-auto}"
SEMANTIC_DTYPE="${SEMANTIC_DTYPE:-float16}"

INIT_FROM="${INIT_FROM:-}"
AUTO_INIT_CKPT="${AUTO_INIT_CKPT:-1}"

# -------------------------- Train hyperparameters -------------------------- #
TRAIN_EPOCHS="${TRAIN_EPOCHS:-11}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-1e-5}"
TRAIN_NUM_WARMUP_UPDATES="${TRAIN_NUM_WARMUP_UPDATES:-4000}"
TRAIN_GRAD_ACCUMULATION_STEPS="${TRAIN_GRAD_ACCUMULATION_STEPS:-1}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-4500}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-$BATCH_SIZE}"
TRAIN_BATCH_SIZE_TYPE="${TRAIN_BATCH_SIZE_TYPE:-frame}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-16}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_SAVE_PER_UPDATES="${TRAIN_SAVE_PER_UPDATES:-10000}"
TRAIN_LAST_PER_UPDATES="${TRAIN_LAST_PER_UPDATES:-10000}"
TRAIN_KEEP_LAST_N_CHECKPOINTS="${TRAIN_KEEP_LAST_N_CHECKPOINTS:--1}"
TRAIN_LOG_SAMPLES="${TRAIN_LOG_SAMPLES:-False}"
TRAIN_LOGGER="${TRAIN_LOGGER:-tensorboard}"
TRAIN_BNB_OPTIMIZER="${TRAIN_BNB_OPTIMIZER:-False}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

STAGE="${STAGE:-4}"
STOP_STAGE="${STOP_STAGE:-6}"

DATA_DIR="$ROOT_DIR/data/$DATASET_NAME"
CSV_WAVS_DIR="$ROOT_DIR/input_csv_wavs/$DATASET_NAME"
CKPT_DIR="$ROOT_DIR/ckpts/SemanticF5TTS_Base_vocos_char_$DATASET_NAME"
EXPECTED_INIT_CKPT="$CKPT_DIR/pretrained_semantic_init.pt"

usage() {
  cat <<'EOF'
Usage:
  ./fine_tune.sh

Default source:
  /home/reg/TTS_DATA/data/YTDT2

Expected SOURCE_PAIRS_DIR format:
  SOURCE_PAIRS_DIR/
  ├── xxx.wav
  ├── xxx.txt
  ├── yyy.wav
  └── yyy.txt

Common env overrides:
  SOURCE_PAIRS_DIR             Folder containing .wav/.txt pairs
  DATASET_NAME                 Dataset name under Custom_TTS/data, default basename of SOURCE_PAIRS_DIR
  PREPARE_DATA=auto|1|0        Build raw.arrow/duration.json from wav/txt pairs
  PREPARE_SEMANTIC=1|0         Run HuBERT semantic GT extraction
  INIT_FROM                    Optional initialized SemanticCFM .pt to copy into this run
  AUTO_INIT_CKPT=1|0           If init ckpt missing, create it automatically
  BASE_VOCAB_PATH              Fixed vocab path
  FIXED_VOCAB_SIZE             Defaults to line count of BASE_VOCAB_PATH
  TRAIN_EPOCHS                 Number of epochs
  TRAIN_LEARNING_RATE          Learning rate
  TRAIN_NUM_WARMUP_UPDATES     Warmup updates
  TRAIN_GRAD_ACCUMULATION_STEPS
  TRAIN_MAX_GRAD_NORM
  BATCH_SIZE                   Alias for TRAIN_BATCH_SIZE_PER_GPU, default 5000
  TRAIN_BATCH_SIZE_PER_GPU     Dynamic frame batch threshold
  TRAIN_BATCH_SIZE_TYPE        frame or sample
  TRAIN_MAX_SAMPLES            Max samples per dynamic batch
  TRAIN_NUM_WORKERS            Dataloader workers
  TRAIN_SAVE_PER_UPDATES       Save model_<update>.pt interval
  TRAIN_LAST_PER_UPDATES       Save model_last.pt interval
  TRAIN_KEEP_LAST_N_CHECKPOINTS
  TRAIN_LOG_SAMPLES=True|False
  TRAIN_LOGGER=tensorboard|wandb|null
  TRAIN_BNB_OPTIMIZER=True|False
  TRAIN_EXTRA_ARGS             Extra Hydra overrides, e.g. 'optim.epochs=3 ckpts.log_samples=False'
  STAGE                        First stage to run, default 0
  STOP_STAGE                   Last stage to run, default 6

Stages:
  0 install/check package and fixed vocab
  1 build csv_wavs metadata + raw.arrow/duration.json
  2 copy fixed vocab + validate prepared dataset
  3 prepare HuBERT semantic ground truth
  4 initialize semantic checkpoint
  5 print dataset/checkpoint summary
  6 start fine-tune

Examples:
  STAGE=0 STOP_STAGE=6 ./fine_tune.sh      Run all
  STAGE=3 STOP_STAGE=3 ./fine_tune.sh      Only rebuild semantic GT
  STAGE=4 STOP_STAGE=6 ./fine_tune.sh      Init checkpoint then train
  STAGE=6 STOP_STAGE=6 ./fine_tune.sh      Train only
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

should_run() {
  local s="$1"
  [[ "$STAGE" -le "$s" && "$STOP_STAGE" -ge "$s" ]]
}

if ! [[ "$STAGE" =~ ^[0-9]+$ && "$STOP_STAGE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: STAGE and STOP_STAGE must be non-negative integers." >&2
  exit 1
fi

if [[ "$STAGE" -gt "$STOP_STAGE" ]]; then
  echo "ERROR: STAGE must be <= STOP_STAGE." >&2
  exit 1
fi

echo "Running stages: STAGE=$STAGE STOP_STAGE=$STOP_STAGE"

if should_run 0; then
  echo "[0/6] Installing Custom_TTS package in editable mode..."
  python -m pip install -e .
else
  echo "[0/6] Skipping package install/check."
fi

if [[ ! -f "$BASE_VOCAB_PATH" ]]; then
  echo "ERROR: BASE_VOCAB_PATH not found: $BASE_VOCAB_PATH" >&2
  exit 1
fi

if [[ -z "$FIXED_VOCAB_SIZE" ]]; then
  FIXED_VOCAB_SIZE="$(wc -l < "$BASE_VOCAB_PATH" | tr -d '[:space:]')"
fi
echo "Using fixed vocab: $BASE_VOCAB_PATH ($FIXED_VOCAB_SIZE tokens)"

if should_run 1; then
  need_prepare=0
  if [[ "$PREPARE_DATA" == "1" ]]; then
    need_prepare=1
  elif [[ "$PREPARE_DATA" == "auto" ]]; then
    if [[ ! -f "$DATA_DIR/raw.arrow" && ! -d "$DATA_DIR/raw" ]]; then
      need_prepare=1
    fi
  elif [[ "$PREPARE_DATA" != "0" ]]; then
    echo "ERROR: PREPARE_DATA must be auto, 1, or 0" >&2
    exit 1
  fi

  if [[ "$need_prepare" == "1" ]]; then
    if [[ ! -d "$SOURCE_PAIRS_DIR" ]]; then
      echo "ERROR: SOURCE_PAIRS_DIR not found: $SOURCE_PAIRS_DIR" >&2
      exit 1
    fi

    echo "[1/6] Building csv_wavs metadata from wav/txt pairs..."
    mkdir -p "$CSV_WAVS_DIR/wavs"
    python - <<PY
from pathlib import Path

src = Path("$SOURCE_PAIRS_DIR").resolve()
out = Path("$CSV_WAVS_DIR").resolve()
metadata = out / "metadata.csv"

pairs = []
for wav in sorted(src.glob("*.wav")):
    txt = wav.with_suffix(".txt")
    if txt.exists():
        text = " ".join(txt.read_text(encoding="utf-8", errors="ignore").split())
        if text:
            pairs.append((wav, text))

if not pairs:
    raise SystemExit(f"No wav/txt pairs found in {src}")

with metadata.open("w", encoding="utf-8") as f:
    f.write("audio_path|text\\n")
    for wav, text in pairs:
        # Absolute paths avoid copying large audio files into input_csv_wavs/wavs.
        f.write(f"{wav.as_posix()}|{text}\\n")

print(f"metadata={metadata}")
print(f"pairs={len(pairs)}")
PY

    echo "[1/6] Preparing F5 raw.arrow and duration.json..."
    python src/custom_tts/train/datasets/prepare_csv_wavs.py "$CSV_WAVS_DIR" "$DATA_DIR" --pretrain
  else
    echo "[1/6] Using existing prepared dataset: $DATA_DIR"
  fi
else
  echo "[1/6] Skipping raw dataset preparation."
fi

if should_run 2; then
  echo "[2/6] Copying fixed vocab and validating prepared dataset..."
  mkdir -p "$DATA_DIR"
  cp "$BASE_VOCAB_PATH" "$DATA_DIR/vocab.txt"
  echo "Copied fixed vocab to $DATA_DIR/vocab.txt"
else
  echo "[2/6] Skipping vocab copy."
fi

if should_run 2 || should_run 3 || should_run 4 || should_run 5 || should_run 6; then
  if [[ ! -f "$DATA_DIR/duration.json" ]]; then
    echo "ERROR: Missing $DATA_DIR/duration.json. Run STAGE=1 first or provide prepared dataset." >&2
    exit 1
  fi
  if [[ ! -f "$DATA_DIR/raw.arrow" && ! -d "$DATA_DIR/raw" ]]; then
    echo "ERROR: Missing $DATA_DIR/raw.arrow or $DATA_DIR/raw. Run STAGE=1 first or provide prepared dataset." >&2
    exit 1
  fi
  if [[ ! -f "$DATA_DIR/vocab.txt" ]]; then
    echo "ERROR: Missing $DATA_DIR/vocab.txt. Run STAGE=2 first or provide vocab." >&2
    exit 1
  fi
fi

if should_run 3; then
  if [[ "$PREPARE_SEMANTIC" == "1" ]]; then
    echo "[3/6] Preparing HuBERT semantic ground truth..."
    custom-tts_prepare-semantic-hubert \
      --dataset-name "$DATASET_NAME" \
      --model-name "$SEMANTIC_MODEL" \
      --layer "$SEMANTIC_LAYER" \
      --batch-size "$SEMANTIC_BATCH_SIZE" \
      --device "$SEMANTIC_DEVICE" \
      --dtype "$SEMANTIC_DTYPE" \
      --overwrite
  else
    echo "[3/6] PREPARE_SEMANTIC=0, skipping semantic preparation."
  fi
else
  echo "[3/6] Skipping semantic preparation."
fi

if should_run 4; then
  echo "[4/6] Ensuring initialized semantic checkpoint exists..."
  mkdir -p "$CKPT_DIR"
  if [[ -n "$INIT_FROM" ]]; then
    if [[ ! -f "$INIT_FROM" ]]; then
      echo "ERROR: INIT_FROM not found: $INIT_FROM" >&2
      exit 1
    fi
    cp "$INIT_FROM" "$EXPECTED_INIT_CKPT"
    echo "Copied initialized checkpoint to $EXPECTED_INIT_CKPT"
  elif [[ ! -f "$EXPECTED_INIT_CKPT" ]]; then
    if [[ "$AUTO_INIT_CKPT" != "1" ]]; then
      echo "ERROR: Missing initialized checkpoint: $EXPECTED_INIT_CKPT" >&2
      echo "Set INIT_FROM=/path/to/pretrained_semantic_init.pt or AUTO_INIT_CKPT=1." >&2
      exit 1
    fi
    custom-tts_init-semantic-ckpt \
      --config-name "$CONFIG_NAME" \
      --dataset-name "$DATASET_NAME" \
      --vocab-size "$FIXED_VOCAB_SIZE"
  else
    echo "Using existing initialized checkpoint: $EXPECTED_INIT_CKPT"
  fi
else
  echo "[4/6] Skipping checkpoint initialization."
fi

if should_run 5 || should_run 6; then
  if [[ ! -f "$EXPECTED_INIT_CKPT" ]]; then
    echo "ERROR: Missing initialized checkpoint: $EXPECTED_INIT_CKPT" >&2
    echo "Run STAGE=4 first, set INIT_FROM, or provide the checkpoint manually." >&2
    exit 1
  fi
fi

if should_run 5; then
  echo "[5/6] Dataset/checkpoint summary..."
  ls -lh "$DATA_DIR"/duration.json "$DATA_DIR"/vocab.txt
  if [[ -f "$DATA_DIR/raw.arrow" ]]; then
    ls -lh "$DATA_DIR/raw.arrow"
  else
    du -sh "$DATA_DIR/raw"
  fi
  ls -lh "$EXPECTED_INIT_CKPT"
else
  echo "[5/6] Skipping summary."
fi

if should_run 6; then
  echo "[6/6] Starting fine-tune..."
  accelerate launch src/custom_tts/train/train.py \
    --config-name "$CONFIG_NAME" \
    datasets.name="$DATASET_NAME" \
    datasets.num_workers="$TRAIN_NUM_WORKERS" \
    datasets.batch_size_per_gpu="$TRAIN_BATCH_SIZE_PER_GPU" \
    datasets.batch_size_type="$TRAIN_BATCH_SIZE_TYPE" \
    datasets.max_samples="$TRAIN_MAX_SAMPLES" \
    optim.epochs="$TRAIN_EPOCHS" \
    optim.learning_rate="$TRAIN_LEARNING_RATE" \
    optim.num_warmup_updates="$TRAIN_NUM_WARMUP_UPDATES" \
    optim.grad_accumulation_steps="$TRAIN_GRAD_ACCUMULATION_STEPS" \
    optim.max_grad_norm="$TRAIN_MAX_GRAD_NORM" \
    optim.bnb_optimizer="$TRAIN_BNB_OPTIMIZER" \
    ckpts.save_per_updates="$TRAIN_SAVE_PER_UPDATES" \
    ckpts.last_per_updates="$TRAIN_LAST_PER_UPDATES" \
    ckpts.keep_last_n_checkpoints="$TRAIN_KEEP_LAST_N_CHECKPOINTS" \
    ckpts.log_samples="$TRAIN_LOG_SAMPLES" \
    ckpts.logger="$TRAIN_LOGGER" \
    model.vocab_size_override="$FIXED_VOCAB_SIZE" \
    $TRAIN_EXTRA_ARGS
else
  echo "[6/6] Skipping fine-tune."
fi
