import geopandas as gpd

# Load the shapefile 
gdf = gpd.read_file("~/data/ava_outlines/outlines2018.shp")

# Preview the data
print(gdf.head())
print(gdf.columns)
print(gdf.shape)

# Reproject to WGS84 (lat/lon)
gdf_wgs84 = gdf.to_crs(epsg=4326)

print(gdf_wgs84.geometry.head())
print(gdf_wgs84.columns)
