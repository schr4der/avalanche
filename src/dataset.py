from torch.utils.data import Dataset
import torch
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.merge import merge
import geopandas as gpd
import math
import os

class GeoTiffSegmentationDataset(Dataset):
    def __init__(self, tile_centers, tile_size, tiff_dir, shp_path, transform=None):
        self.tile_centers = tile_centers  # list of (row, col)
        self.tiff_dir = tiff_dir
        self.shapefile = gpd.read_file(shp_path)
        self.transform = transform
        self.tile_size = tile_size

    def __len__(self):
        return len(self.tile_centers)

    def __getitem__(self, idx):
        row, col = self.tile_centers[idx]
        
        # --- Load stitched image and metadata ---
        image, meta = stitch_tiles(row, col, self.tiff_dir, self.tile_size)

        # --- Rasterize shapefile to match the image extent ---
        mask = rasterize_shp(self.shapefile, meta)

        # Normalize and convert to torch
        img_tensor = torch.from_numpy(image).float()  # shape: [C, H, W]
        mask_tensor = torch.from_numpy(mask).long()   # shape: [H, W]

        # Apply any custom transform (augmentations etc.)
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

def get_tile_filename(row, col, base_dir):
    return os.path.join(base_dir, f"swissalti3d_2024_{row}-{col}_0.5_2056_5728.tif")

def stitch_tiles(center_row, center_col, base_dir, n_tiles):
    tiles = []
    
    # Loop over nxn neighborhood
    offset = math.floor(n_tiles/2)
    for dr in range(-offset, offset+1):
        for dc in range(-offset, offset+1):
            r = center_row + dr
            c = center_col + dc
            filepath = get_tile_filename(r, c, base_dir)
            if os.path.exists(filepath):
                src = rasterio.open(filepath)
                tiles.append(src)
            else:
                print(f"Missing tile: {filepath}")

    if not tiles:
        raise ValueError("No tiles found.")

    # Merge tiles
    mosaic, out_transform = merge(tiles)

    # Use metadata from the first tile
    out_meta = tiles[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # Close all files
    for src in tiles:
        src.close()

    return mosaic, out_meta

def rasterize_shp(shapefile, raster_meta):
    gdf = shapefile.to_crs(raster_meta['crs'])
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(raster_meta['height'], raster_meta['width']),
        transform=raster_meta['transform'],
        fill=0,
        dtype='uint8'
    )
    return mask

#Use like:
#Todo train/val/test split
from torch.utils.data import DataLoader

tile_centers = [(2594,1128)]
train_dataset = GeoTiffSegmentationDataset(tile_centers, 3, "../data/topo_maps/", "../data/ava_outlines/outlines2018.shp")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
print(next(iter(train_loader)))