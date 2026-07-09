import geopandas as gpd
import pandas as pd

mask2019 = gpd.read_file('LiDAR2019_polyg.shp')
mask2020 = gpd.read_file('LiDAR2020_polyg.shp')
mask2021 = gpd.read_file('LiDAR2021_polyg.shp')
mask2022 = gpd.read_file('LiDAR2022_polyg.shp')

mask2019['source'] = 'LiDAR2019'
mask2020['source'] = 'LiDAR2020'
mask2021['source'] = 'LiDAR2021'
mask2022['source'] = 'LiDAR2022'
united_gdf = pd.concat([mask2019, mask2020, mask2021, mask2022], ignore_index=True)
united_gdf.to_file('lidar_mask_polygs.shp')

# Verify the result
print(united_gdf)