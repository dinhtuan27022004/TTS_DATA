# Custom TTS

This project is a semantic-conditioned F5-TTS variant copied from
`/home/reg/TTS_DATA/F5-TTS-Vietnamese`.

The added path follows the "way 2" design:

```text
Text tokens ---------> original F5 text embedding ----\
                                                       DiT + CFM -> mel -> vocoder -> audio
Text tokens -> TextToSemantic -> semantic condition --/
Reference audio/mel -> original F5 cond mel ----------/
```

Use `src/custom_tts/configs/SemanticF5TTS_Base.yaml` as the starting config.
If your dataset contains a frame-level `semantic` field, training uses:

```text
loss = L_flow + semantic.loss_weight * L_semantic
```

If `semantic` is absent, the model still runs with normal F5 flow matching and
uses the predicted semantic condition as an extra learned conditioning branch.

## Prepare HuBERT Semantic Ground Truth

After preparing an F5-style dataset under `data/<dataset_name>`, add semantic
targets with:

```bash
custom-tts_prepare-semantic-hubert \
  --dataset-name <dataset_name> \
  --batch-size 4 \
  --overwrite
```

By default this uses `facebook/hubert-base-ls960` and writes a new `raw.arrow`
with an extra `semantic` field. The previous `raw.arrow` or `raw/` dataset is
backed up with a `.before_semantic` suffix.

Then train:

```bash
accelerate launch src/custom_tts/train/train.py \
  --config-name SemanticF5TTS_Base.yaml \
  datasets.name=<dataset_name>
```
