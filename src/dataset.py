from torch.utils.data import Dataset
import torch
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.merge import merge
import geopandas as gpd
import torch.nn.functional as F
import math
import os
import re
import glob

class GeoTiffSegmentationDataset(Dataset):
    def __init__(self, tile_size, step_size, tiff_dir, shp_path, transform=None):
        self.tiff_dir = tiff_dir
        self.shapefile = gpd.read_file(shp_path)
        self.transform = transform
        self.tile_size = tile_size
        self.step_size = step_size
        filenames = [f for f in os.listdir(tiff_dir) if os.path.isfile(os.path.join(tiff_dir, f))]
        # print(filenames)
        self.tile_centers = []  # list of (row, col)
        for filename in filenames:
            match = re.search(r'_(\d+-\d+)_', filename)
            if match:
                coords = match.group(1).split('-')
                x = int(coords[0])
                y = int(coords[1])
                self.tile_centers.append((x, y))
            else:
                print(filename)
                print("error: unrecognized tif file format")

        
        
        # Convert to a set for fast lookup
        coord_set = set(self.tile_centers)

        # Define relative offsets for 8 neighbors
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),         ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
        print("before")
        # Keep only the coordinates where all 8 neighbors are present
        # TODO generalize to nxn
        self.tile_centers = [
            (x, y)
            for (x, y) in self.tile_centers
            if all((x + dx, y + dy) in coord_set for dx, dy in neighbor_offsets)
        ]
        self.tile_centers = list(filter(
            lambda pair: pair[0] % self.step_size == 0 and pair[1] % self.step_size == 0,
            self.tile_centers
        ))


    def __len__(self):
        return len(self.tile_centers)

    def __getitem__(self, idx):
        row, col = self.tile_centers[idx]
        
        # --- Load stitched image and metadata ---
        image, meta = stitch_tiles(row, col, self.tiff_dir, self.tile_size)
        image = normalize(image)
        # --- Rasterize shapefile to match the image extent ---
        mask = rasterize_shp(self.shapefile, meta)

        # Normalize and convert to torch
        img_tensor = torch.from_numpy(image).float()  # shape: [C, H, W]
        mask_tensor = torch.from_numpy(mask).long()   # shape: [H, W]

        # Apply any custom transform (augmentations etc.)
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        img_tensor, original_size = pad_to_multiple(img_tensor, 16)
        mask_tensor, _ = pad_to_multiple(mask_tensor.unsqueeze(0), 16)
        mask_tensor = mask_tensor.squeeze(0)

        return img_tensor, mask_tensor

#TODO improve to mean/std dataset stats img = (img - mean) / std
def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def get_tile_filename(row, col, base_dir):
    pattern = os.path.join(base_dir, f"swissalti3d_*_{row}-{col}_2_2056_5728.tif")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found for tile {row}-{col}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for tile {row}-{col}: {matches}")
    return matches[0]

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

def pad_to_multiple(tensor, multiple):
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    return F.pad(tensor, padding, mode='constant', value=0), (h, w)

# #Use like:
# #Todo train/val/test split
# from torch.utils.data import DataLoader

# # tile_centers = [(2594,1128)]
# train_dataset = GeoTiffSegmentationDataset(3, 1, "../data/topo_maps/swiss_topo/", "../data/ava_outlines/outlines2018.shp")
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# print(next(iter(train_loader)))
