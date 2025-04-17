from torch.utils.data import Dataset
import torch
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.merge import merge
import geopandas as gpd
import math
import os
import re

class GeoTiffSegmentationDataset(Dataset):
    def __init__(self, tile_size, step_size, tiff_dir, shp_path, transform=None):
        self.tiff_dir = tiff_dir
        self.shapefile = gpd.read_file(shp_path)
        self.transform = transform
        self.tile_size = tile_size
        self.step_size = step_size
        filenames = [f for f in os.listdir(tiff_dir) if os.path.isfile(os.path.join(tiff_dir, f))]

        self.tile_centers = []  # list of (row, col)
        for filename in filenames:
            match = re.search(r'_(\d{4}-\d{4})_', filename)
            if match:
                coords = match.group(1).split('-')
            else:
                print("error: unrecognized tif file format")
            x = int(coords[0])
            y = int(coords[1])
            if x%self.step_size == 0 and y%self.step_size==0:
                self.tile_centers.append((x, y))
        print(self.tile_centers)
        
        # Convert to a set for fast lookup
        coord_set = set(self.tile_centers)
        print(coord_set)

        # Define relative offsets for 8 neighbors
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),         ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]

        # Keep only the coordinates where all 8 neighbors are present
        self.tile_centers = [
            (x, y)
            for (x, y) in self.tile_centers
            if all((x + dx, y + dy) in coord_set for dx, dy in neighbor_offsets)
        ]

        print(self.tile_centers)


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
    return os.path.join(base_dir, f"swissalti3d_2024_{row}-{col}_2_2056_5728.tif")

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

# tile_centers = [(2594,1128)]
train_dataset = GeoTiffSegmentationDataset(3, 1, "../data/topo_maps/swiss_topo/", "../data/ava_outlines/outlines2018.shp")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
print(next(iter(train_loader)))
