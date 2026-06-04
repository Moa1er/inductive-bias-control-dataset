import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification

# import the newly renamed class
from generate_dataset import PatchPermutedCIFAR


# apply transforms on the fly since we are wrapping an existing dataset
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img) if self.transform else img, label


def get_loaders(batch_size=32, patch_size=8, is_vit=False):
    # base dataset needs to output tensors for the native pytorch patch permuter to work
    base_transform = transforms.ToTensor()
    raw_val = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=base_transform
    )

    perm_val = PatchPermutedCIFAR(raw_val, patch_size=patch_size)

    # downstream transforms (skip ToTensor since it is already a tensor)
    if is_vit:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        )

    clean_loader = DataLoader(
        ApplyTransform(raw_val, transform), batch_size=batch_size, num_workers=2
    )
    perm_loader = DataLoader(
        ApplyTransform(perm_val, transform), batch_size=batch_size, num_workers=2
    )

    return clean_loader, perm_loader


def evaluate(model, loader, device, max_batches=50):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_batches:
                break

            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)

            # extract logits if it's a huggingface model
            logits = out.logits if hasattr(out, "logits") else out
            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return (correct / total) * 100


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}\n")

    # 1. resnet evaluation
    print("loading resnet20:")
    resnet = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
    ).to(device)
    cnn_clean, cnn_perm = get_loaders(is_vit=False)

    print("evaluating resnet:")
    cnn_acc_clean = evaluate(resnet, cnn_clean, device)
    cnn_acc_perm = evaluate(resnet, cnn_perm, device)

    # 2. vit evaluation
    print("\nloading vit-base:")
    vit = AutoModelForImageClassification.from_pretrained(
        "nateraw/vit-base-patch16-224-cifar10"
    ).to(device)
    vit_clean, vit_perm = get_loaders(is_vit=True)

    print("evaluating vit:")
    vit_acc_clean = evaluate(vit, vit_clean, device)
    vit_acc_perm = evaluate(vit, vit_perm, device)

    # final results
    print("\nresults summary:")
    print(
        f"resnet20 | clean: {cnn_acc_clean:.1f}% | permuted: {cnn_acc_perm:.1f}% | delta: {cnn_acc_perm - cnn_acc_clean:.1f}%"
    )
    print(
        f"vit-b/16 | clean: {vit_acc_clean:.1f}% | permuted: {vit_acc_perm:.1f}% | delta: {vit_acc_perm - vit_acc_clean:.1f}%"
    )
