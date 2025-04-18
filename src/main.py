from dataset import GeoTiffSegmentationDataset
from model import UNet
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
        num_workers = 1

    # Load Dataset 
    generator = Generator().manual_seed(421)
    tile_centers = [(2594,1128)]
    dataset = GeoTiffSegmentationDataset(3, 3, "../data/swiss_topo_v1/swiss_topo/", "../data/ava_outlines/outlines2018.shp")



    # Split test/train/val
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    # Setup Model
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 1

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

    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS = 10

    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

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
            
            dc = dice_coefficient(y_pred, mask)
            loss = criterion(y_pred, mask)
            
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
        
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                mask = mask.unsqueeze(1)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = dice_coefficient(y_pred, mask)
                
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

    # Saving the model
    torch.save(model.state_dict(), 'my_checkpoint.pth')

if __name__ == "__main__":
    main()