import argparse
import os
from pathlib import Path
from unet import *
import torch
import logging
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import Polygon
import numpy as np
from affine import Affine
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
import rasterio.features
from shapely.geometry import shape
import json
import datetime

def get_intf_coords(intf_name):
    intf_dict_file = open('intf_coord.json', 'r')
    intf_coords = json.load(intf_dict_file)
    x0 = intf_coords[intf_name]['east']
    y0 = intf_coords[intf_name]['north']
    dx = intf_coords[intf_name]['dx']
    dy = intf_coords[intf_name]['dy']
    nlines = intf_coords[intf_name]['nlines']
    ncells = intf_coords[intf_name]['ncells']
    lidar_mask = intf_coords[intf_name]['lidar_mask']
    num_nonz_p = intf_coords[intf_name]['nonz_num']
    bo = intf_coords[intf_name]['byte_order']

    x4000 = x0 + 4000*dx
    x8500 = x4000 + 4500*dx

    return (x0, y0, dx, dy,ncells, nlines, x4000, x8500,lidar_mask,num_nonz_p,bo)
def get_intf_lidar_mask(intf_name):
    with open('lidar_intf_mask.txt', 'r') as f:
        mask = 'no_mask'
        for line in f:
            if intf_name[:8] == line[8:16] and intf_name[9:17] == line[24:32]:
                mask = line[40:49]
            elif intf_name[:8] == line[8:16]:
                mask = line[40:49]


    return mask
def patchify(input_array, window_size, stride, mask_array= None,nonz_pathces = True):
    if mask_array is not None:
        assert input_array.shape == mask_array.shape, "Mask array should be the same shape as input array"
        mask_array = mask_array[:, 4000:8500]


    input_array = input_array[:, 4000:8500]
    rows, cols = input_array.shape
    data_patches,mask_patches,data_patches_nonz,mask_patches_nonz = [],[],[],[]

    for i in range(0, rows - window_size[0] + 1, stride[0]):
      data_row , mask_row = [],[]
      for j in range(0, cols - window_size[1] + 1, stride[1]):
         data_patch = input_array[i:i + window_size[0], j:j + window_size[1]]
         data_row.append(data_patch)
         if mask_array is not None:
            mask_patch = mask_array[i:i + window_size[0], j:j + window_size[1]]
            if mask_patch.any() !=0 and nonz_pathces:
                data_patches_nonz.append(data_patch)
                mask_patches_nonz.append(mask_patch)

      data_patches.append(data_row)
      mask_patches.append(mask_row)
    if mask_array is None:
       return np.array(data_patches)
    else:
        return np.array(data_patches),np.array(mask_patches), np.array(data_patches_nonz), np.array(mask_patches_nonz)

def reconstruct_intf_prediction(data, intf_coords, net,patch_size, stride, rth, add_lidar_mask = True, plot = False):
    x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nonz_p,bo = intf_coords
    patch_H,patch_W = patch_size
    data = (data + np.pi) / (2 * np.pi)

    reconstructed_pred = np.zeros((data.shape[0] * patch_H // stride + patch_H * (1 - 1 // stride),
                                   data.shape[1] * patch_W // stride + patch_W * (1 - 1 // stride)))

    if add_lidar_mask:
        lidar_mask_df = gpd.read_file('lidar_mask_polygs.shp')
        mask_polyg = lidar_mask_df[lidar_mask_df['source'] == intf_lidar_mask]
        height, width = reconstructed_pred.shape[0], reconstructed_pred.shape[1]  # Raster dimensions
        transform = rasterio.transform.from_origin(x4000, y0, dx, dy)  # Example transform

        # Create an empty raster array
        raster = np.zeros((height, width), dtype=np.uint8)
        geometries = mask_polyg['geometry'].tolist()
        # Rasterize the polygons
        rasterized_polygon = rasterize(
            [(g, 1) for g in geometries],  # List of (geometry, value) tuples
            out_shape=raster.shape,
            transform=transform,
            fill=0,
            all_touched=True,  # If True, all pixels touched by geometries will be burned in
            dtype=raster.dtype
        )

        if plot:
            plt.imshow(rasterized_polygon)
            plt.show()
    for i in range(data.shape[0]):
        print(i)

        for j in range(data.shape[1]):

            is_within_mask = True
            if add_lidar_mask:
                is_within_mask = np.all(rasterized_polygon[i * patch_H // stride:i * patch_H // stride + patch_H,
                                        j * patch_W // stride: j * patch_W // stride + patch_W])

            if is_within_mask:

                image = torch.tensor(data[i, j]).unsqueeze(0).unsqueeze(1).to(device=device,
                                                                              dtype=torch.float32,
                                                                              memory_format=torch.channels_last)
                pred = net(image)
                pred = (F.sigmoid(pred) > 0.5).float()
                pred = pred.squeeze(1).squeeze(0).cpu().detach().numpy()

                reconstructed_pred[i * patch_H // stride:i * patch_H // stride + patch_H,
                j * patch_W // stride: j * patch_W // stride + patch_W] += pred / stride ** 2


    reconstructed_pred = np.where(reconstructed_pred > rth, 1, 0).astype(np.float32)

    return reconstructed_pred

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

def get_args():
    parser = argparse.ArgumentParser(description='Predict subsidence polygons for new interferogram')
    parser.add_argument('--intfs_dir',  type=str, default='./', help='interferogram path')
    parser.add_argument('--intfs_list', type=str, default='a list of intfs in format YYYYMMDD_YYYYMMDD seperated by comma')
    parser.add_argument('--model_dir', type=str,default='./models/')
    parser.add_argument('--model_file', type=str, help='the .pth model file')
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=(200,100), help='patch H, patch W. These have to match the patch size in training')
    parser.add_argument('--strdpp',type = int, default = 2, help='strides per patch in both directions')
    parser.add_argument('--plot_polygs',  type=bool, default=False)
    parser.add_argument('--output_polygs_dir', type=str)
    parser.add_argument('--rth',type = float, default=0.25)
    parser.add_argument('--add_gt_polygs', type=bool, default=False)
    parser.add_argument('--gt_polygons_file_path', type=str, default='sub_20231001.shp')
    return parser.parse_args()


if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    args = get_args()
    stride = (args.patch_size[0]//args.strdpp, args.patch_size[1]//args.strdpp)
    intfs_list = args.intfs_list.split(',')
    input_files = []
    for intf in intfs_list:
        for filename in os.listdir(args.intfs_dir):
            if filename.endswith('.unw') and intf[:8] == filename[9:17] and intf[9:] == filename[25:33]:
                input_files.append(filename)
                break
    input_file_paths = [args.intfs_dir + filename for filename in input_files]
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(args.model_dir + args.model_file, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.eval()
    logging.info('Model loaded!')

    for i,p in enumerate(input_file_paths):

        intf = intfs_list[i]
        intfs_coords = get_intf_coords(intf)
        x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nz,bo = intfs_coords

        data = np.fromfile(p, dtype=np.float32, count=-1, sep='', offset=0).reshape(nlines, ncells)
        if bo == 'MSBFirst':
            data = data.byteswap().newbyteorder('<')
        patchified_intf = patchify(data, args.patch_size, stride, mask_array=None, nonz_pathces=False)

        reconstructed_pred = reconstruct_intf_prediction(patchified_intf,intfs_coords,net,args.patch_size,args.strdpp,args.rth)
        polygons = plg_indx2longlat(mask_array_to_polygons(reconstructed_pred),intfs_coords)

        if not os.path.exists(Path(args.output_polygs_dir)):
            os.makedirs(args.output_polygs_dir)
        out_polyg_f = args.output_polygs_dir + intf +'_predicted_polyogns.shp'
        polygons.to_file(out_polyg_f)


        if args.add_gt_polygs:
            gdf = gpd.read_file(args.gt_polygon_file_path)
            start_date = intf[:8]
            end_date =  intf[9:]

            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            gt_polygs = gdf[(gdf['start_date']==start_date) & (gdf['end_date']==end_date)]
            if gt_polygs.empty:
                print('No ground truth for intf ' + intf + '. Skipping...')
        data = data[:,4000:8500]
        extent = [x4000, x4000 + dx * data.shape[1], y0 - dy * data.shape[0], y0]

        if args.plot_polygs:
            plt.imshow(data, extent=extent)
            list_of_polygons = polygons.geometry.tolist()

            for p in list_of_polygons:
                x, y = p.exterior.xy
                plt.plot(x, y,color='red')
            if args.add_gt_polygs and not gt_polygs.empty:
                list_of_gt_polygs = gt_polygs.geometry.tolist()
                for gp in list_of_gt_polygs:
                    xg, yg = gp.exterior.xy
                    plt.plot(xg, yg,color='blue')

            plt.show()



