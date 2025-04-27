import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.features import rasterize
from rasterio.merge import merge
from pyproj import Transformer
import numpy as np
import math
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from dataset import GeoTiffSegmentationDataset

# Utility script to visualize masks overlaid over topographic images from the dataset
def main():
    n_tiles = 25

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("using cuda")
        num_workers = torch.cuda.device_count() * 1
        torch.cuda.empty_cache()
    else:
        num_workers = 4

    dataset = GeoTiffSegmentationDataset(3, 3, "../data/swiss_topo_v2/swiss_topo/", "../data/ava_outlines/outlines2018.shp")
    dataloader = DataLoader(dataset=dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=1,
                                shuffle=True)

    for idx, img_mask in enumerate(tqdm(dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        img = img
        print(img.shape, mask.shape)
        # Remove extra dimensions
        img_np = img.squeeze().cpu().numpy()      # shape: (H, W) .squeeze()
        mask_np = mask.squeeze().cpu().numpy()    # shape: (H, W)
        img_np = img_np[0]
        # Create a masked array
        masked_mask = np.ma.masked_where(mask_np == 0, mask_np)

        # Plot
        plt.figure(figsize=(10, 10))
        plt.title("TIFF with Colored Mask Overlay")

        plt.imshow(img_np, cmap="gray")                    # grayscale image
        plt.imshow(masked_mask, cmap="spring", alpha=0.6)  # mask overlay
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
