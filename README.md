# Patch permutation control dataset

This repository contains the generation code for the patch permutation control dataset (PPCD). This dataset is designed to evaluate the spatial inductive biases of vision models by destroying macro-geometric structure while keeping local textures intact.

## Dataset generation mechanism
The generation script downloads the CIFAR-10 validation set, slices each $32 \times 32$ image into non-overlapping $8 \times 8$ patches, randomly permutes their grid coordinates, and saves the resulting images into class-segregated directories. 

## Repository structure
* `generate_dataset.py` - the core script handling image block tokenization, random permutation mapping, and physical file serialization.
* `requirements.txt` - project dependencies.

## Setup and installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Moa1er/inductive-bias-control-dataset.git](https://github.com/Moa1er/inductive-bias-control-dataset.git)
   cd inductive-bias-control-dataset
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generating the dataset

To physically generate the control dataset images and create the deployable `.zip` archive, run:
```bash
python generate_dataset.py
```

### Core generation logic snippet

The permutation mapping relies on slicing the underlying image arrays along their spatial axes without changing the inner pixel configurations:

```python
# break down image into a 5d array: (channels, num_h, p, num_w, p)
patches = img_np.reshape(c, num_patches_h, p, num_patches_w, p)

# reorder axes to group patch blocks together
patches = patches.transpose(1, 3, 0, 2, 4)
flattened_patches = patches.reshape(num_patches_h * num_patches_w, c, p, p)

# generate a unique permutation and reassemble the layout grid
perm = np.random.permutation(len(flattened_patches))
shuffled_patches = flattened_patches[perm]
```