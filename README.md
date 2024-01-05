# Stable Video Diffusion + ControlNet

## Requirement

- Python 3.10.13
- pip install -r requirements.txt

## Quick Start

### Multi Machine Training

```sh
make train_dist NUM_NODES=n_nodes CONFIG=path/to/config.yaml
```

### Signle Machine Training

```sh
make train CONFIG=path/to/config.yaml
```

### Check Available Machine

```sh
make check
```

### Debug

```sh
make debug
```
