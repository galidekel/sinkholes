import os
import geopandas as gpd
from get_intf_info import *
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def plg_indx2longlat(polyg_gdf, intf_coords):
    x4000, y0, dx, dy = intf_coords[5], intf_coords[1], intf_coords[2], intf_coords[3]
    polyg_list = polyg_gdf['geometry'].tolist()
    polyg_longlat = []
    for polyg in polyg_list:
        new_coords = [(x4000 + x * dx, y0 - y * dy) for x, y in polyg.exterior.coords]
        polyg = Polygon(new_coords)
        polyg_longlat.append(polyg)

    polyg_longlat_gdf = gpd.GeoDataFrame(geometry=polyg_longlat, crs='EPSG:4326')
    return polyg_longlat_gdf


# List all files and directories in the specified path
directory_path = 'pred_outputs/job_train_11days_preset_by_intf_20_05_13h45_2024-05-24_13h35checkpoint_epoch60/polygs/'
outpath = directory_path + 'polygs_longlat/'
os.makedirs(outpath, exist_ok=True)
file_list = os.listdir(directory_path)
for file in file_list:
    if file.endswith('.shp'):
        polyg_gdf = gpd.read_file(directory_path + file)
        intf = file[:17]
        intf_coords = get_intf_coords(intf)
        polyg_longlat = plg_indx2longlat(polyg_gdf,intf_coords)
        polyg_longlat.to_file(outpath+intf+'_predicted_polygons_longlat.shp')


        polygs_longlat_gdf = gpd.read_file(outpath+intf+'_predicted_polygons_longlat.shp')
        polygs_longlat_gdf.plot()
        plt.show()


