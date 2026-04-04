# External Model Assets

This directory is reserved for upstream model repositories and checkpoints that are required to regenerate embeddings.

Tracked metadata:

- `model_sources.lock.json`: upstream repository URLs and pinned commits

Expected layout:

```text
external/
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

Setup flow:

1. Run `bash scripts/setup_external_repos.sh`
2. Place any missing checkpoints in the paths listed above
3. Run `bash scripts/check_external_assets.sh`

Notes:

- The upstream repositories are not committed to this repository.
- Large checkpoints should stay untracked under `external/`.
- `roberta` embeddings do not use `external/`; they are loaded through `transformers`.

