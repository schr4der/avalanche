from dataset import GeoTiffSegmentationDataset
from model import UNet
from mini_CNN import SimpleFCN
from utils import dice_coefficient

from torch.utils.data import DataLoader, random_split
from torch import Generator, nn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

def main():
    # Setup Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("using cuda")
        num_workers = torch.cuda.device_count() * 1
        torch.cuda.empty_cache()
    else:
        num_workers = 4

    # Load Dataset 
    generator = Generator().manual_seed(421)
    tile_centers = [(2594,1128)]
    dataset = GeoTiffSegmentationDataset(3, 3, "../data/swiss_topo_v2/swiss_topo/", "../data/ava_outlines/outlines2018.shp")
    # Split test/train/val
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    # Setup Model
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 2

    train_dataloader = DataLoader(dataset=train_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=4, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor([3]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    EPOCHS = 10

    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []
    test_losses = []
    test_dcs = []

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        
        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            mask = mask.unsqueeze(1)
            
            y_pred = model(img)

            _, _, h_tgt, w_tgt = mask.shape
            y_pred = y_pred[:, :, :h_tgt, :w_tgt]
            optimizer.zero_grad()
            loss = criterion(y_pred, mask)

            dc = dice_coefficient(torch.sigmoid(y_pred), mask)
            
            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)
        
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0
        # Report validation loss
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                mask = mask.unsqueeze(1)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = dice_coefficient(torch.sigmoid(y_pred), mask)
                
                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)
        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)

        # Test evaluation
        test_running_loss = 0
        test_running_dc = 0

        model.eval()
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                mask = mask.unsqueeze(1)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = dice_coefficient(torch.sigmoid(y_pred), mask)

                test_running_loss += loss.item()
                test_running_dc += dc.item()

            test_loss = test_running_loss / (idx + 1)
            test_dc = test_running_dc / (idx + 1)

        test_losses.append(test_loss)
        test_dcs.append(test_dc)

        print(f"Test Loss EPOCH {epoch + 1}: {test_loss:.4f}")
        print(f"Test DICE EPOCH {epoch + 1}: {test_dc:.4f}")
        print("-" * 30)



    # Saving the model
    torch.save(model.state_dict(), 'my_checkpoint.pth')

    # Plot Loss and Dice over Epochs
    epochs = range(1, EPOCHS + 1)

    # Plot Losses
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()

    # Plot Dice Coefficients
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_dcs, label='Train Dice', marker='o')
    plt.plot(epochs, val_dcs, label='Validation Dice', marker='o')
    plt.plot(epochs, test_dcs, label='Test Dice', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Training, Validation, and Test Dice Coefficient')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dice_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
