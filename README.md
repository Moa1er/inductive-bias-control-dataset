# Patch permutation control dataset

This repository contains the generation and evaluation code for the patch permutation control dataset (PPCD). This dataset is designed to evaluate the spatial inductive biases of vision models by destroying macro-geometric structure while keeping local textures intact.

## Dataset generation mechanism
The generation script downloads the CIFAR-10 validation set, slices each $32 \times 32$ image into non-overlapping $8 \times 8$ patches, randomly permutes their grid coordinates, and saves the resulting images into class-segregated directories. 

## Repository structure
* `generate_dataset.py` - the core script handling image block tokenization, random permutation mapping, and physical file serialization.
* `evaluate.py` - evaluates pre-trained ResNet and Vision Transformer (ViT) models on clean vs. patch-permuted CIFAR-10 images.
* `requirements.txt` - project dependencies.

## Setup and installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Moa1er/inductive-bias-control-dataset.git
   cd inductive-bias-control-dataset
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generating the dataset

To physically generate the control dataset images and create the deployable `.zip` archive, run:
```bash
python generate_dataset.py
```

### 2. Evaluating models

To evaluate pre-trained models (ResNet20 and Vision Transformer) on clean and patch-permuted images, run:
```bash
python evaluate.py
```

### Core generation logic snippet

The permutation mapping relies on slicing PyTorch tensors along their spatial axes:

```python
# slice image into grid blocks natively in pytorch
patches = img.reshape(c, h // p, p, w // p, p).permute(1, 3, 0, 2, 4)
patches = patches.reshape(-1, c, p, p)

# shuffle patches deterministically based on image index
torch.manual_seed(seed + idx)
perm = torch.randperm(patches.size(0))
shuffled = patches[perm]

# rebuild the image tensor
grid = shuffled.reshape(h // p, w // p, c, p, p)
out_img = grid.permute(2, 0, 3, 1, 4).reshape(c, h, w)
```