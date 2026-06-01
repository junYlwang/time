# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This is a **two-stage time-series foundation model**: a neural codec compresses raw series into discrete tokens, then a pretrained LLM (Qwen3-4B) is fine-tuned via LoRA to predict these tokens autoregressively.

**Stage 1 — Codec (ConvNeXt autoencoder + RFSQ quantization):**
- Encoder: ConvNeXt blocks with 8× temporal downsampling (`down_ratio: [2,2,2]`)
- Quantizer: RFSQ (Residual Finite Scalar Quantization), 2 layers × 1000-token codebook = 2000 total tokens
- Decoder: symmetric ConvNeXt with 8× upsampling
- Input: `[B, 1, 4096]` raw time series → output: `[B, 2, 512]` code indices in latent space

**Stage 2 — LLM Next-Token Prediction:**
- Frozen codec encoder tokenizes raw series into code indices
- Tokens are interleaved per quantizer: `[q0_t0, q1_t0, q0_t1, q1_t1, ...]`
- Qwen3-4B + LoRA (r=16, all linear projections) predicts next token
- 2000 special tokens `<ts_0>` through `<ts_1999>` added to vocabulary
- Input: `[B, 512]` raw series (left-padded, min 504 valid points)

## Directory Layout

```
time/
  src/
    modules/       # Core model components
      encoder_wo_quantize.py   # ConvNeXt encoder (no quantizer)
      decoder.py               # ConvNeXt decoder
      quantizer.py             # RFSQQuantizer, RVQQuantizer, NoQuantizer
      stochastic_fsq.py        # FSQ / RFSQ implementation
      vq.py                    # RVQ with EMA codebooks
      codec_token_ntp.py       # CodecTokenNTPModel — wraps frozen codec + trainable LLM
      predictor.py             # Transformer-based codec token predictor (Stage 1.5)
      revin.py                 # ReversibleInstanceNorm1D (no learnable params)
      backbones.py             # ConvNeXtBlock, Down/UpSamplingBlock, GRN
      loss.py                  # Multi-scale log-magnitude STFT loss
      probe.py                 # Linear/MLP/Transformer probes + Transformer building blocks
      decomposition.py         # Trend/residual decomposition
      utils.py                 # AttrDict, checkpoint I/O, coverage tracking
    datasets/
      time_moe_dataset.py       # TimeMoEDataset — loads BinaryDataset/GeneralDataset
      binary_dataset.py         # Time-300B binary format (.bin + meta.json)
      general_dataset.py        # .json/.npy/.pkl format
      time_codec_dataset.py     # SplitTimeSeriesCodecDataset — random segment sampling
      llm_codec_dataset.py      # SplitRawSeriesDataset — left-padded segments for LLM training
  tools/                         # Training and inference entry points
    codec_train.py               # Stage 1: codec reconstruction training
    codec_prediction_train.py    # Stage 1.5: codec + predictor training
    codec_prediction_pad_train.py # Variant with padding-aware loss
    llm_train.py                 # Stage 2: LLM NTP training with HuggingFace Trainer
    codec_infer.py               # Codec-only inference on Time-300B test subsets
    codec_infer_window.py        # Sliding-window codec inference
    codec_infer_gift_eval_shortest_m4_yearly.py  # Codec-only reconstruction eval on M4-Yearly
  configs/
    data_v1.json                 # Time-300B split manifest (train/valid/test paths)
    codec-base-rfsq2-data-v1.yaml
    codec-prediction-rfsq2-data-v1.yaml
    llm-train.yaml               # LLM training config + gift_eval section
    codec-infer-gift-eval-m4-yearly-shortest.yaml
  checkpoint/                    # Saved model checkpoints
    ts-codec-rfsq-2               # Codec only
    ts-codec-rfsq-2-prediction    # Codec + predictor
    512-ts-codec-rfsq-2-prediction # 512-segment codec + predictor (used by LLM training)
  scripts/                       # rjob/torchrun submission scripts
  data/
    gift_eval/data.py            # Local copy of GIFT-Eval Dataset class
```

The gift-eval integration code lives in a **separate repo** at `/mnt/shared-storage-user/wangjunyi/gift-eval/`:
- `src/gift_eval/codec_llm.py` — `CodecLLMPredictor` implementing GluonTS predictor interface
- `cli/codec_llm.py` — CLI to run full GIFT-Eval benchmark evaluation

## Common Commands

### Training

```bash
# Stage 1: Train codec (reconstruction)
cd /mnt/shared-storage-user/wangjunyi/time
torchrun --nproc_per_node=4 tools/codec_train.py --config configs/codec-base-rfsq2-data-v1.yaml

# Stage 1.5: Train codec + predictor
torchrun --nproc_per_node=4 tools/codec_prediction_train.py --config configs/codec-prediction-rfsq2-data-v1.yaml

# Stage 2: Train LLM (requires codec checkpoint)
torchrun --nproc_per_node=4 tools/llm_train.py --config configs/llm-train.yaml
```

### Codec-only inference (on Time-300B test subsets)

```bash
python tools/codec_infer.py --config configs/codec-base-rfsq2-data-v1.yaml
# Config must have `inference.test_split`, `inference.checkpoint_path`, `inference.output_dir`
```

### GIFT-Eval benchmark evaluation

```bash
cd /mnt/shared-storage-user/wangjunyi/gift-eval
python cli/codec_llm.py --llm-config /mnt/shared-storage-user/wangjunyi/time/configs/llm-train.yaml
```

All evaluation parameters are read from the `gift_eval` section in `llm-train.yaml`:
- `gift_eval.adapter_path` — LoRA adapter directory (output of Stage 2 training)
- `gift_eval.data_root` — local path to GIFT-Eval datasets
- `gift_eval.output_dir` — where `all_results.csv` + `config.json` are written
- `gift_eval.num_samples` — number of forecast trajectories (default 20)
- `gift_eval.temperature` / `top_p` / `top_k` — LLM sampling parameters
- `--limit N` runs only the first N datasets (useful for smoke testing)
- `--output-dir`, `--num-samples`, `--model-name` can override config values

## Key Conventions

**Normalization:** Codec was trained with RevIN on full 4096-length segments (no padding). LLM training/inference uses per-sample masked z-score that only considers valid (non-padding) time steps. This is equivalent for unpadded data since RevIN has no learnable affine parameters.

**Left-padding:** The model left-pads to multiples of `downsample_factor=8`. Masked normalization zeros out padding positions and normalizes valid positions only. This ensures the encoder sees normalized data for valid steps and zeros for padding.

**Token ordering:** Codec tokens from two quantizers are interleaved: `[q0_t0, q1_t0, q0_t1, q1_t1, ...]`. During autoregressive generation, `AlternatingTimeTokenMask` enforces this pattern by restricting each step's logits to the correct quantizer's token range.

**Time-300B data format:** BinaryDataset uses `.bin` files with `meta.json` containing per-sequence offsets/lengths. Despite dataset names containing `_with_missing`, Time-300B preprocessed all missing values — no NaN exists in training data. GIFT-Eval datasets DO contain NaN and require the `_impute()` method in `CodecLLMPredictor`.

**Checkpoints:** `load_checkpoint()` in `utils.py` loads with `torch.load(..., map_location=device)`. Checkpoints are flat dicts with keys like `encoder`, `quantizer`, `decoder`, `input_norm`, `predictor`.

**AttrDict:** Config YAML is loaded into `AttrDict` (a dict subclass with `__dict__ = self`). Access fields with both dot notation (`h.latent_dim`) and dict notation (`h["latent_dim"]`). Nested YAML sections (like `gift_eval`) become plain dicts, not AttrDicts — convert manually: `AttrDict(h.get("gift_eval", {}))`.

## Coding Style

**Minimal abstraction.** Do not extract a function unless at least one of these holds:
- The same logic is genuinely reused elsewhere (not just hypothetically).
- The boundary between two distinct logical stages within a single function is sharp enough that extracting and naming the stage makes the parent function easier to follow.
- Giving the logic a name significantly clarifies what a non-obvious block does.

When in doubt, keep the code inline. Inline code that is read top-to-bottom is preferred over jumping through many small single-caller helpers.

**No defensive or future-proof programming.** Write code for the concrete data, model architecture, and workflow that exists today:
- Do not add fallback paths for quantizer types, checkpoint formats, or data layouts that are not currently in use.
- Do not add parameters or configuration knobs for hypothetical future use cases.
- Validate inputs only when a mistake is both plausible and hard to debug; skip guards against scenarios that the existing codebase never produces.
- Hardcoded paths and constants are acceptable when they reflect the fixed operating environment of this project.
