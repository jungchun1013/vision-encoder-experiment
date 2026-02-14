# Vision Encoder Experiments

A tiny pipline to run experiments on vision encoders

## Models
- CLIP
- DINO
- DINOv2
- MAE
- ResNet
- SigLIP
- ViT-supervised
- I-JEPA (requried download)
- SimCLR (requried download)


## Project Structure
```
  vision-encoder/
  ├── models/          # encoder loading & config
  │   ├── mae.py
  │   ├── clip.py
  │   ├── dino.py
  │   ├── ...
  │   └── registry.py  # unified model registry
  ├── wrappers/        # normalize encoder outputs to common interface
  │   └── encoder.py   # BaseEncoder with extract_features(), embed()
  ├── tasks/           # downstream evaluation
  │   ├── linear_probe.py
  │   ├── knn.py
  │   ├── finetune.py
  │   └── retrieval.py
  ├── datasets/        # dataset loading helpers
  │   └── loader.py
  ├── main.py          # entry point
  └── pyproject.toml
```

## Downstream task / Visualization (TODO)
- k-nn (clustering)
- Linear probe
- Progressing Masking
- t-sne