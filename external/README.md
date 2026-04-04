# External Model Assets

This directory is reserved for upstream model repositories and checkpoints that are required to regenerate embeddings. These are not committed to this repository due to size.

Tracked metadata:

- `model_sources.lock.json`: upstream repository URLs and pinned commits

## Models

| Model | Type | Parameters | Hidden dim | Layers | Upstream repository | Checkpoint |
|-------|------|-----------|------------|--------|-------------------|------------|
| **RNA-FM** | RNA language model (masked LM) | ~99M | 640 | 12 | [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM) | `RNA-FM_pretrained.pth` |
| **ERNIE-RNA** | RNA language model (motif-level masking) | ~86M | 768 | 12 | [Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA) | `ERNIE-RNA_pretrain.pt` |
| **RNABERT** | RNA BERT (masked LM + structural alignment) | ~0.5M | 120 | 6 | [mana438/RNABERT](https://github.com/mana438/RNABERT) | `bert_mul_2.pth` |
| **RiNALMo** | RNA language model (650M scale) | ~650M | 1280 | 33 | [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo) | `rinalmo_giga_pretrained.pt` |
| **RoBERTa** | General-purpose masked LM (not RNA-specific) | ~125M | 768 | 12 | [Hugging Face](https://huggingface.co/roberta-base) | Downloaded automatically via `transformers` |
| **One-hot** | Non-learned baseline | — | 4 | 1 | — | — |

## Expected layout

```text
external/
├── model_sources.lock.json
├── RNA-FM/
│   └── RNA-FM_pretrained.pth
├── ERNIE-RNA/
│   ├── checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt
│   └── src/dict/
├── RNABERT/
│   ├── MLM_SFP.py
│   ├── RNA_bert_config.json
│   └── bert_mul_2.pth
└── RiNALMo/
    └── weights/rinalmo_giga_pretrained.pt
```

## Setup

```bash
# Clone upstream repos at pinned commits
bash scripts/setup_external_repos.sh

# Place any missing checkpoints in the paths listed above

# Validate all assets are present
bash scripts/check_external_assets.sh
```

## Notes

- RoBERTa does not use `external/`; it is loaded through `transformers` from Hugging Face.
- One-hot encoding requires no external dependencies.
- Large checkpoints should stay untracked under `external/`.
- See `model_sources.lock.json` for exact pinned commits.
