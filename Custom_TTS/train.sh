#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET_NAME="${DATASET_NAME:-my_dataset}"
INPUT_DIR="${INPUT_DIR:-}"
CONFIG_NAME="${CONFIG_NAME:-SemanticF5TTS_Base.yaml}"

PREPARE_DATA="${PREPARE_DATA:-auto}"
PREPARE_MODE="${PREPARE_MODE:-pretrain}"
PREPARE_WORKERS="${PREPARE_WORKERS:-}"

PREPARE_SEMANTIC="${PREPARE_SEMANTIC:-1}"
SEMANTIC_MODEL="${SEMANTIC_MODEL:-facebook/hubert-base-ls960}"
SEMANTIC_LAYER="${SEMANTIC_LAYER:--1}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-4}"
SEMANTIC_DEVICE="${SEMANTIC_DEVICE:-auto}"
SEMANTIC_DTYPE="${SEMANTIC_DTYPE:-float16}"

INIT_SEMANTIC_CKPT="${INIT_SEMANTIC_CKPT:-0}"
INIT_PRETRAINED="${INIT_PRETRAINED:-}"

TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-12000}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-16}"
BASE_VOCAB_PATH="${BASE_VOCAB_PATH:-/home/reg/TTS_DATA/models/f5-tts-v0/vocab.txt}"
USE_BASE_VOCAB="${USE_BASE_VOCAB:-1}"
FIXED_VOCAB_SIZE="${FIXED_VOCAB_SIZE:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

DATA_DIR="$ROOT_DIR/data/$DATASET_NAME"

usage() {
  cat <<'EOF'
Usage:
  DATASET_NAME=my_dataset INPUT_DIR=/path/to/csv_wavs ./train.sh

Expected INPUT_DIR format for prepare step:
  INPUT_DIR/
  ├── metadata.csv   # rows: audio_path|text
  └── wavs/

Common env overrides:
  DATASET_NAME                 Dataset folder name under Custom_TTS/data
  INPUT_DIR                    Raw csv_wavs dataset path. If omitted, existing data/<DATASET_NAME> is used
  PREPARE_DATA=auto|1|0        auto prepares only if data files are missing
  PREPARE_MODE=pretrain|finetune
  PREPARE_SEMANTIC=1|0         Run HuBERT/WavLM semantic extraction
  INIT_SEMANTIC_CKPT=1|0       Create pretrained_semantic_init.pt before training
  INIT_PRETRAINED              Optional F5 checkpoint path/url for semantic init
  SEMANTIC_MODEL               HF model, default facebook/hubert-base-ls960
  SEMANTIC_BATCH_SIZE          Teacher extraction batch size
  TRAIN_BATCH_SIZE_PER_GPU     F5 frame batch threshold
  TRAIN_MAX_SAMPLES            Max samples per dynamic batch
  BASE_VOCAB_PATH              Vocab file used to infer fixed model vocab size
  USE_BASE_VOCAB=1|0           Copy BASE_VOCAB_PATH to data/<dataset>/vocab.txt before train
  FIXED_VOCAB_SIZE             Optional fixed model vocab size. Defaults to line count of BASE_VOCAB_PATH
  TRAIN_EXTRA_ARGS             Extra Hydra overrides, e.g. 'optim.epochs=3 ckpts.log_samples=False'
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "[1/5] Installing Custom_TTS package in editable mode..."
python -m pip install -e .

need_prepare=0
if [[ "$PREPARE_DATA" == "1" ]]; then
  need_prepare=1
elif [[ "$PREPARE_DATA" == "auto" ]]; then
  if [[ ! -f "$DATA_DIR/raw.arrow" && ! -d "$DATA_DIR/raw" ]]; then
    need_prepare=1
  fi
fi

if [[ "$need_prepare" == "1" ]]; then
  if [[ -z "$INPUT_DIR" ]]; then
    echo "ERROR: INPUT_DIR is required because prepared dataset was not found at: $DATA_DIR" >&2
    usage
    exit 1
  fi

  echo "[2/5] Preparing F5 dataset at $DATA_DIR ..."
  prepare_args=()
  if [[ "$PREPARE_MODE" == "pretrain" ]]; then
    prepare_args+=(--pretrain)
  elif [[ "$PREPARE_MODE" != "finetune" ]]; then
    echo "ERROR: PREPARE_MODE must be pretrain or finetune" >&2
    exit 1
  fi
  if [[ -n "$PREPARE_WORKERS" ]]; then
    prepare_args+=(--workers "$PREPARE_WORKERS")
  fi
  python src/custom_tts/train/datasets/prepare_csv_wavs.py "$INPUT_DIR" "$DATA_DIR" "${prepare_args[@]}"
else
  echo "[2/5] Using existing prepared dataset: $DATA_DIR"
fi

if [[ ! -f "$DATA_DIR/duration.json" ]]; then
  echo "ERROR: Missing $DATA_DIR/duration.json. Run prepare step first." >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/vocab.txt" ]]; then
  echo "ERROR: Missing $DATA_DIR/vocab.txt. Use PREPARE_MODE=pretrain or provide vocab.txt manually." >&2
  exit 1
fi

if [[ "$USE_BASE_VOCAB" == "1" ]]; then
  if [[ ! -f "$BASE_VOCAB_PATH" ]]; then
    echo "ERROR: BASE_VOCAB_PATH not found: $BASE_VOCAB_PATH" >&2
    exit 1
  fi
  cp "$BASE_VOCAB_PATH" "$DATA_DIR/vocab.txt"
  echo "Copied base vocab to $DATA_DIR/vocab.txt"
fi

if [[ -z "$FIXED_VOCAB_SIZE" ]]; then
  if [[ ! -f "$BASE_VOCAB_PATH" ]]; then
    echo "ERROR: BASE_VOCAB_PATH not found: $BASE_VOCAB_PATH" >&2
    exit 1
  fi
  FIXED_VOCAB_SIZE="$(wc -l < "$BASE_VOCAB_PATH" | tr -d '[:space:]')"
  echo "Using fixed vocab size from $BASE_VOCAB_PATH: $FIXED_VOCAB_SIZE"
fi

if [[ "$PREPARE_SEMANTIC" == "1" ]]; then
  echo "[3/5] Preparing HuBERT semantic ground truth..."
  custom-tts_prepare-semantic-hubert \
    --dataset-name "$DATASET_NAME" \
    --model-name "$SEMANTIC_MODEL" \
    --layer "$SEMANTIC_LAYER" \
    --batch-size "$SEMANTIC_BATCH_SIZE" \
    --device "$SEMANTIC_DEVICE" \
    --dtype "$SEMANTIC_DTYPE" \
    --overwrite
else
  echo "[3/5] Skipping semantic preparation."
fi

echo "[4/6] Verifying dataset files..."
ls -lh "$DATA_DIR"/duration.json "$DATA_DIR"/vocab.txt "$DATA_DIR"/raw.arrow

if [[ "$INIT_SEMANTIC_CKPT" == "1" ]]; then
  echo "[5/6] Initializing semantic architecture checkpoint..."
  init_args=(
    --config-name "$CONFIG_NAME"
    --dataset-name "$DATASET_NAME"
  )
  if [[ -n "$INIT_PRETRAINED" ]]; then
    init_args+=(--pretrained "$INIT_PRETRAINED")
  fi
  if [[ -n "$FIXED_VOCAB_SIZE" ]]; then
    init_args+=(--vocab-size "$FIXED_VOCAB_SIZE")
  fi
  custom-tts_init-semantic-ckpt "${init_args[@]}"
else
  echo "[5/6] Skipping explicit semantic checkpoint initialization."
fi

echo "[6/6] Starting training..."
accelerate launch src/custom_tts/train/train.py \
  --config-name "$CONFIG_NAME" \
  datasets.name="$DATASET_NAME" \
  datasets.batch_size_per_gpu="$TRAIN_BATCH_SIZE_PER_GPU" \
  datasets.max_samples="$TRAIN_MAX_SAMPLES" \
  ${FIXED_VOCAB_SIZE:+model.vocab_size_override=$FIXED_VOCAB_SIZE} \
  $TRAIN_EXTRA_ARGS
