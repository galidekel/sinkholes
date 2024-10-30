import os

from get_intf_info import *
from shapely.geometry import Polygon

import numpy as np
from affine import Affine
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import shapes

import rasterio.features
from shapely.geometry import shape, polygon, box

import geopandas as gpd


def mask_array_to_polygons(mask_array):
    transform = Affine.identity()    # Generate polygons from the mask array
    polygons = shapes(mask_array, transform=transform)
    shapely_polygons = [shape(geom) for geom, value in polygons if value == 1]
    polygons_gpd = gpd.GeoDataFrame(geometry=shapely_polygons,crs="EPSG:4326")
    return polygons_gpd

def plg_indx2longlat(polyg_gdf, intf_coords):
    x4000, y0, dx, dy = intf_coords[6], intf_coords[1], intf_coords[2], intf_coords[3]
    polyg_list = polyg_gdf['geometry'].tolist()
    polyg_longlat = []
    for polyg in polyg_list:
        new_coords = [(x4000 + x * dx, y0 - y * dy) for x, y in polyg.exterior.coords]
        polyg = Polygon(new_coords)
        polyg_longlat.append(polyg)

    polyg_longlat_gdf = gpd.GeoDataFrame(geometry=polyg_longlat, crs='EPSG:4326')
    return polyg_longlat_gdf
if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    directory_path = 'pred_outputs2/job_train_11ddiffs_rand_by_patch_2024-08-11_22h33checkpoint_epoch125/job_10_29_18h26/'
    dir_polygs = directory_path #+ 'polygs/'
    outpath = dir_polygs + 'polygs_longlat/'
    os.makedirs(outpath, exist_ok=True)
    file_list = os.listdir(dir_polygs)
    for file in file_list:
        if file.endswith('.shp'):
            polyg_gdf = gpd.read_file(dir_polygs + file)
            intf = file[:17]
            intf_coords = get_intf_coords(intf)
            polyg_longlat = plg_indx2longlat(polyg_gdf,intf_coords)
            polyg_longlat.to_file(outpath+intf+'_predicted_polygons_longlat.shp')


            polygs_longlat_gdf = gpd.read_file(outpath+intf+'_predicted_polygons_longlat.shp')
            image = np.load( directory_path + intf +'_image.npy', allow_pickle=True)
            fig, ax1 = plt.subplots(1, 1)
            x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nz = get_intf_coords(intf)

            extent = [x4000, x8500, y0 - dy * nlines, y0]



            #ax1.imshow(image,cmap='jet')
            polyg_longlat.plot(ax=ax1, facecolor='none', edgecolor='black')

        plt.show()


