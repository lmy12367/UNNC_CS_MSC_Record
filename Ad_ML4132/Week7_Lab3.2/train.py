import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from net import UNet

def dice_loss(pred, target, epsilon=1e-6):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return 1 - (2. * intersection + epsilon) / (union + epsilon)

class NucleiDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_ids = sorted(os.listdir(os.path.join(root_dir, "stage1_train")))
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.root_dir, 'stage1_train', img_id, 'images', f'{img_id}.png')
        img = Image.open(img_path).convert('RGB')
        mask_dir = os.path.join(self.root_dir, 'stage1_train', img_id, 'masks')
        mask_files = os.listdir(mask_dir)
        combined_mask = None
        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            single_mask = np.array(Image.open(mask_path).convert('L'))
            if combined_mask is None:
                combined_mask = single_mask
            else:
                combined_mask = np.maximum(combined_mask, single_mask)
        mask = Image.fromarray(combined_mask)
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0)
        return img, mask

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

root_dir = "./data/data"
max_train_samples = 512
max_val_samples = 128

train_dataset = NucleiDataset(root_dir, transform=transform, mask_transform=mask_transform, max_samples=max_train_samples)
val_dataset = NucleiDataset(root_dir, transform=transform, mask_transform=mask_transform, max_samples=max_val_samples)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, train_loader, val_loader, num_epochs=1):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_dices, val_dices = [], []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss, correct_train, total_train = 0, 0, 0
        running_train_dice = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct_train += (preds == masks).sum().item()
            total_train += masks.numel()
            running_train_dice += (2 * torch.sum(preds * masks) + 1e-6) / (torch.sum(preds) + torch.sum(masks) + 1e-6)

        train_loss = running_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_dice = running_train_dice / len(train_loader)

        model.eval()
        running_val_loss, correct_val, total_val = 0, 0, 0
        running_val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.unsqueeze(1)

                outputs = model(images)
                loss = dice_loss(outputs, masks)
                running_val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_val += (preds == masks).sum().item()
                total_val += masks.numel()
                running_val_dice += (2 * torch.sum(preds * masks) + 1e-6) / (torch.sum(preds) + torch.sum(masks) + 1e-6)

        val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_dice = running_val_dice / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dices.append(train_dice.cpu().numpy())
        val_dices.append(val_dice.cpu().numpy())

        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {val_dice:.4f}")

    return train_losses, val_losses, train_accs, val_accs, train_dices, val_dices

train_losses, val_losses, train_accs, val_accs, train_dices, val_dices = train(model, train_loader, val_loader,
                                                                               num_epochs=5)

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/unet_model.pth")

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Train vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")
plt.title("Train vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(train_dices, label="Train Dice")
plt.plot(val_dices, label="Val Dice")
plt.title("Train vs Validation Dice Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Dice Coefficient")
plt.legend()

plt.tight_layout()
plt.savefig("checkpoints/training_curves.png")
plt.show()

with open("checkpoints/results.txt", "w") as f:
    f.write("===== Final Results =====\n")
    f.write(f"Train Accuracy: {train_accs[-1]:.4f}\n")
    f.write(f"Val Accuracy:   {val_accs[-1]:.4f}\n")
    f.write(f"Train Dice:     {train_dices[-1]:.4f}\n")
    f.write(f"Val Dice:       {val_dices[-1]:.4f}\n")
