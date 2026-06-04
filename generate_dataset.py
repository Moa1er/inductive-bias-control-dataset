import os
import shutil
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# for reproducibility
torch.manual_seed(42)


class PatchPermutedCIFAR(torch.utils.data.Dataset):
    def __init__(self, base_dataset, patch_size=8, seed=42):
        self.dataset = base_dataset
        self.patch_size = patch_size
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        c, h, w = img.shape
        p = self.patch_size

        # slice image into grid blocks natively in pytorch
        patches = img.reshape(c, h // p, p, w // p, p).permute(1, 3, 0, 2, 4)
        patches = patches.reshape(-1, c, p, p)

        # shuffle patches deterministically based on image index
        torch.manual_seed(self.seed + idx)
        perm = torch.randperm(patches.size(0))
        shuffled = patches[perm]

        # rebuild the image tensor
        grid = shuffled.reshape(h // p, w // p, c, p, p)
        out_img = grid.permute(2, 0, 3, 1, 4).reshape(c, h, w)

        return out_img, label


def save_comparison(orig, perm):
    # save a quick before/after check
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    axes[0].imshow(orig.permute(1, 2, 0).numpy())
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(perm.permute(1, 2, 0).numpy())
    axes[1].set_title("permuted")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("placeholder.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    patch_size = 8
    out_dir = "control_dataset"

    # grab raw cifar10 as tensors
    transform = torchvision.transforms.ToTensor()
    base_val = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    permuted_val = PatchPermutedCIFAR(base_val, patch_size)

    # generate the sample visual
    save_comparison(base_val[0][0], permuted_val[0][0])

    # clear out old directory if it exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print(f"generating dataset in {out_dir}")

    # dump images into class folders
    for idx in range(len(permuted_val)):
        img, label = permuted_val[idx]
        class_name = base_val.classes[label]

        folder = os.path.join(out_dir, class_name)
        os.makedirs(folder, exist_ok=True)

        save_image(img, os.path.join(folder, f"perm_{idx}.png"))

    # zip it up for hosting
    shutil.make_archive(f"{out_dir}_patch{patch_size}", "zip", out_dir)
    print("done!")
