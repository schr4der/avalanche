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

def get_tile_filename(row, col, base_dir):
    return os.path.join(base_dir, f"swissalti3d_2024_{row}-{col}_2_2056_5728.tif")

def stitch_tiles(center_row, center_col, base_dir):
    tiles = []
    
    # Loop over 3x3 neighborhood
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

n_tiles = 25

# Load raster
src, metadata = stitch_tiles(2594, 1128, "../data/topo_maps/swiss_topo/")
transform = metadata['transform']
raster_crs = metadata['crs']
raster_shape = (metadata["height"], metadata["width"])

# Load and reproject shapefile
gdf = gpd.read_file("../data/ava_outlines/outlines2018.shp")
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

# Create mask
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=raster_shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# Create a masked array to hide the background (0s)
masked_mask = np.ma.masked_where(mask == 0, mask)

# Plot
plt.figure(figsize=(10, 10))
plt.title("TIFF with Bright Colored Mask Overlay")

# Show TIFF in grayscale
plt.imshow(src[0], cmap="gray")

# Overlay the mask in bright green (you can change the colormap)
plt.imshow(masked_mask, cmap="spring", alpha=0.6)

plt.axis("off")
plt.show()
