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
from shapely.geometry import box
from scipy.ndimage import gaussian_filter
import time
from tqdm import tqdm

class GeoTiffSegmentationDataset(Dataset):
    def __init__(self, tile_size, step_size, tiff_dir, shp_path, transform=None):
        self.tiff_dir = tiff_dir
        self.shapefile = gpd.read_file(shp_path)
        self.transform = transform
        self.tile_size = tile_size
        self.step_size = step_size
        self.mean = np.array([2.01592310e+03, 9.13464615e-04, 5.35885466e-02, -5.61953177e-10])
        self.std = np.array([7.20663286e+02, 8.06257248e-04, 1.80814646e+00, 3.35383448e-04])
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


        # sum_ = 0
        # sum_sq = 0
        # count = 0
        # self.mean = np.zeros(4)
        # self.std = np.zeros(4)

        # # --- Mean/Std on Raw Data (Layer 0 only) ---
        # for row, col in tqdm(self.tile_centers):
        #     image, meta = stitch_tiles(row, col, self.tiff_dir, self.tile_size)  # image shape: (1, H, W)

        #     sum_ += image.sum(axis=(1, 2))         # shape (1,)
        #     sum_sq += (image ** 2).sum(axis=(1, 2))  # shape (1,)
        #     count += image.shape[1] * image.shape[2]

        # self.mean[0] = sum_ / count
        # self.std[0] = np.sqrt((sum_sq / count) - self.mean[0] ** 2)

        # print("mean (raw):", self.mean)
        # print("std  (raw):", self.std)

        # # --- Mean/Std for Derived Features (Layers 1–3) ---
        # sum_der = np.zeros(3)
        # sum_sq_der = np.zeros(3)
        # count_der = 0

        # for row, col in tqdm(self.tile_centers):
        #     image, meta = stitch_tiles(row, col, self.tiff_dir, self.tile_size)

        #     img_normalized = normalize_with_stats(image, self.mean[0:1], self.std[0:1])
        #     blurred_image = gaussian_filter(img_normalized, sigma=(0, 1, 1))

        #     # Compute features, take only layers 1:4 (slope, aspect, curvature, etc.)
        #     image_stack = compute_terrain_features(blurred_image, cell_size=meta['transform'][0])[1:4]  # shape: (3, H, W)

        #     sum_der += image_stack.sum(axis=(1, 2))  # shape: (3,)
        #     sum_sq_der += (image_stack ** 2).sum(axis=(1, 2))  # shape: (3,)
        #     count_der += image_stack.shape[1] * image_stack.shape[2]

        # self.mean[1:4] = sum_der / count_der
        # self.std[1:4] = np.sqrt((sum_sq_der / count_der) - self.mean[1:4] ** 2)

        # print("mean (full):", self.mean)
        # print("std  (full):", self.std)


    def __len__(self):
        return len(self.tile_centers)

    def __getitem__(self, idx):
        # start_time = time.time()
        row, col = self.tile_centers[idx]
        
        # --- Load stitched image and metadata ---
        image, meta = stitch_tiles(row, col, self.tiff_dir, self.tile_size)
        img_normalized = normalize_with_stats(image, self.mean[0:1], self.std[0:1])
        
        blurred_image = gaussian_filter(img_normalized, sigma=(0, 1, 1))
        # --- Compute terrain features ---
        image_stack = compute_terrain_features(blurred_image, cell_size = meta['transform'][0])
        image_stack[1:4] = normalize_with_stats(image_stack[1:4], self.mean[1:4], self.std[1:4])
        # img_tensor = image_stack
        img_tensor = torch.from_numpy(image_stack).float()


        # --- Rasterize shapefile to match the image extent ---
        mask = rasterize_shp(self.shapefile, meta)
        # Normalize and convert to torch
        mask_tensor = torch.from_numpy(mask).long()   # shape: [H, W]
        mask_bool = mask_tensor.bool()  # Make sure it's boolean (or binary 0/1)

        # num_ones = mask_bool.sum().item()
        # num_pixels = mask_bool.numel()
        # num_zeros = num_pixels - num_ones

        # print(f"Number of 1s: {num_ones}")
        # print(f"Number of 0s: {num_zeros}")
        # Apply any custom transform (augmentations etc.)
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        img_tensor, original_size = pad_to_multiple(img_tensor, 16)
        mask_tensor, _ = pad_to_multiple(mask_tensor.unsqueeze(0), 16)
        mask_tensor = mask_tensor.squeeze(0)
            
        # print(f"Shape Img: {img_tensor.shape}")
        # print(f"Shape Mask: {mask_tensor.shape}")

        # --- Downsample both image and mask ---
        scale = 0.5  # or any float < 1.0

        # Downsample image: [C, H, W] -> [1, C, H, W] -> interpolate -> [C, h, w]
        #img_tensor = F.interpolate(img_tensor.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)
        #img_tensor = img_tensor.squeeze(0)

        ## Downsample mask: [H, W] -> [1, 1, H, W] -> interpolate -> [h, w]
        #mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(), scale_factor=scale, mode='nearest')
        #mask_tensor = mask_tensor.squeeze(0).squeeze(0).long()

        # end_time = time.time()
        # print(f"Execution time: {end_time - start_time:.4f} seconds")
        return img_tensor, mask_tensor

    #TODO calculate mean/std as [C]
def normalize_with_stats(stack, mean, std, eps=1e-6):
    return (stack - mean[:, None, None]) / (std[:, None, None] + eps)

def compute_terrain_features(dem, cell_size=1.0):
    """
    dem: np.ndarray of shape [H, W] with elevation values
    Returns: slope, aspect, curvature as np.ndarrays of shape [H, W]
    """
    if dem.ndim == 3:
        dem = np.squeeze(dem)  # Convert [1, H, W] → [H, W]
    dzdx = np.gradient(dem, axis=1) / cell_size
    dzdy = np.gradient(dem, axis=0) / cell_size

    slope = np.sqrt(dzdx**2 + dzdy**2)
    aspect = np.arctan2(-dzdy, dzdx)  # negative dzdy so north is 0

    # Curvature: simplified as Laplacian
    curvature = np.gradient(dzdx, axis=1) + np.gradient(dzdy, axis=0)

    return np.stack([dem, slope, aspect, curvature], axis=0)

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

    bounds = rasterio.transform.array_bounds(
        raster_meta['height'], raster_meta['width'], raster_meta['transform']
    )
    raster_box = box(*bounds)

    gdf =  gdf[gdf.intersects(raster_box)]
    
    if gdf.empty:
        return np.zeros((raster_meta['height'], raster_meta['width']), dtype='uint8')

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
