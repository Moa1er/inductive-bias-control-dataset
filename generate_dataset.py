import os
import shutil
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

# set seed for absolute reproducibility across runs
np.random.seed(42)
torch.manual_seed(42)

class PatchPermutedDataset(Dataset):
    """
    wraps a dataset and randomizes the grid positions of its patches
    """
    def __init__(self, base_dataset, patch_size=8):
        self.base_dataset = base_dataset
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # convert tensor to numpy for block slicing operations
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img)
            
        c, h, w = img_np.shape
        p = self.patch_size
        
        num_patches_h = h // p
        num_patches_w = w // p
        
        # break down image into a 5d array: (c, num_h, p, num_w, p)
        patches = img_np.reshape(c, num_patches_h, p, num_patches_w, p)
        # reorder axes to isolate patch blocks: (num_h, num_w, c, p, p)
        patches = patches.transpose(1, 3, 0, 2, 4)
        # flatten grid into a sequence of items: (num_patches, c, p, p)
        flattened_patches = patches.reshape(num_patches_h * num_patches_w, c, p, p)
        
        # generate a unique permutation for this sample instance
        perm = np.random.permutation(len(flattened_patches))
        shuffled_patches = flattened_patches[perm]
        
        # reassemble the permuted blocks back into a grid shape
        shuffled_grid = shuffled_patches.reshape(num_patches_h, num_patches_w, c, p, p)
        # restore native axis order: (c, num_h, p, num_w, p)
        reassembled = shuffled_grid.transpose(2, 0, 4, 1, 3)
        # flatten back to target dimensions: (c, h, w)
        reassembled = reassembled.reshape(c, h, w)
        
        return torch.tensor(reassembled, dtype=torch.float32), label

def generate_static_dataset(output_dir="control_dataset", patch_size=8):
    # keep raw scale transformation for clean file visualization saving
    transform = transforms.Compose([transforms.ToTensor()])
    base_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    class_names = base_val.classes
    
    # wrap dataset with patch scrambler logic
    permuted_stream = PatchPermutedDataset(base_val, patch_size=patch_size)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for name in class_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
        
    print(f"generating permuted images into {output_dir}...")
    
    for idx in range(len(permuted_stream)):
        img_tensor, label = permuted_stream[idx]
        class_name = class_names[label]
        img_path = os.path.join(output_dir, class_name, f"permuted_{idx}.png")
        save_image(img_tensor, img_path)
        
    # zip the folder for external distribution upload hosting
    zip_name = f"{output_dir}_patch{patch_size}"
    shutil.make_archive(zip_name, 'zip', output_dir)
    print(f"saved complete archive to: {zip_name}.zip")

if __name__ == "__main__":
    generate_static_dataset(patch_size=8)
