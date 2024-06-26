import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_score import multiclass_dice_coeff, dice_coeff

import numpy as np
from pathlib import Path

from affine import Affine
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import shapes

import rasterio.features
from shapely.geometry import shape, polygon, box

import geopandas as gpd
from unet import *

from os import listdir

import json

import os

import logging
from datetime import datetime
from lidar_mask import *

def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg

def get_pred_args():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')
    parser.add_argument('--full_intf_dir',  type=str, default='./', help='full interferogram path')
    parser.add_argument('--input_patch_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='patches inputs')
    parser.add_argument('--plot_data',  type=bool, default=False)
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--eleven_days_diff',  type=str, default='True', help='Flag to take only 11 days difference interferograms')
    parser.add_argument('--intf_list', type=str, default = None, help='a list of intf ids divided by comma')

    parser.add_argument('--model', '-m', default='checkpoint_epoch28.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

    parser.add_argument('--data_stride', type=int, default=2)
    parser.add_argument('--recon_th', type=float, default=0.5)

    parser.add_argument('--valset_from_partition', type=str, default=None, help='val set from a partition_File')
    parser.add_argument('--job_name', type=str, default='job', help='unique job name')
    parser.add_argument('--plot', type=str,default='False')


    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Set the logging level (e.g., INFO)
    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)
    now = datetime.now().strftime("%m_%d_%Hh%M")

    args = get_pred_args()
    args.eleven_days_diff = str2bool(args.eleven_days_diff)
    args.plot = str2bool(args.plot)
    if not is_running_locally:
        args.plot = False
    job_name = args.job_name +'_'+ now
    model_name = args.model.split('.')[0]
    output_path = 'pred_outputs/' + model_name + '/' + job_name + '/'
    try:
        os.makedirs(output_path)
        logging.info(f"Directory '{output_path}' created successfully")
    except FileExistsError:
        logging.info(f"Directory '{output_path}' already exists")
    except Exception as e:
        logging.info(f"An error occurred: {e}")
    log_file = output_path + args.job_name + '_' + now + '.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)

    logging.info('Running job {} with model {}.pth and args: {}'.format(job_name, model_name, args))


    net = UNet(n_channels=1, n_classes=1, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load('./models/'+args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.eval()
    logging.info('Model loaded!')



    patch_H, patch_W = args.patch_size
    data_dir = args.input_patch_dir + 'data_patches_H' + str(patch_H) + '_W' + str(patch_W)+'_strpp{}'.format(args.data_stride) + ('_11days' if args.eleven_days_diff else '')
    mask_dir = args.input_patch_dir + 'mask_patches_H' + str(patch_H) + '_W' + str(patch_W) + '_strpp{}'.format(args.data_stride) + ('_11days' if args.eleven_days_diff else '')
    if args.valset_from_partition is not None:
        with open(args.valset_from_partition, 'r') as file:
            loaded_data = json.load(file)
            intf_list = loaded_data['val']
            logging.info('taking test intfs from partition {}. test list: {}'.format(args.valset_from_partition,intf_list))

    elif args.intf_list is not None:
        intf_list = args.intf_list.split(',')
    else:
        intf_list = [file[13:30] for file in listdir(data_dir) if 'nonz' not in file]

    lidar_mask_df = gpd.read_file('lidar_mask_polygs.shp')

    for intf in intf_list:


        x0,y0,dx,dy,x4000,x8500,intf_lidar_mask = get_intf_coords(intf)

        mask_polyg = lidar_mask_df[lidar_mask_df['source'] == intf_lidar_mask]

        if args.plot:
            plt.rcParams['backend'] = 'Qt5Agg'
            mask_polyg.plot()
            plt.show()

        data_file_name = 'data_patches_' + intf + '_H' + str(patch_H) + '_W' + str(patch_W)+'_strpp{}'.format(args.data_stride) +'.npy'
        mask_file_name = 'mask_patches_' + intf + '_H' + str(patch_H) + '_W' + str(patch_W)+'_strpp{}'.format(args.data_stride) +'.npy'
        data_path = data_dir + '/' + data_file_name
        mask_path = mask_dir + '/' + mask_file_name
        data = np.load(data_path)
        data = (data + np.pi) /( 2*np.pi)
        mask = np.load(mask_path)
        assert data.ndim == 4 and mask.ndim == 4, "number of input dims should be 4 got data: {} mask: {} instead".format(data.ndim,mask.ndim)
        reconstructed_intf = np.zeros((data.shape[0] * patch_H // args.data_stride + patch_H * (1-1 // args.data_stride),data.shape[1] * patch_W // args.data_stride + patch_W*(1-1 // args.data_stride)))
        reconstructed_mask = np.zeros((data.shape[0] * patch_H // args.data_stride + patch_H * (1-1 // args.data_stride),data.shape[1] * patch_W // args.data_stride + patch_W * (1-1 // args.data_stride)))
        reconstructed_pred = np.zeros((data.shape[0] * patch_H // args.data_stride + patch_H * (1-1 // args.data_stride),data.shape[1] * patch_W // args.data_stride + patch_W* (1-1 // args.data_stride)))

        for i in range(data.shape[0]):
            print(i)

            for j in range(data.shape[1]):
                reconstructed_intf[i * patch_H // args.data_stride: i * patch_H // args.data_stride + patch_H,j * patch_W // args.data_stride: j * patch_W // args.data_stride + patch_W] += data[i, j] / args.data_stride ** 2
                reconstructed_mask[i * patch_H // args.data_stride:i * patch_H // args.data_stride + patch_H,
                j * patch_W // args.data_stride: j * patch_W // args.data_stride + patch_W] += mask[
                                                                                                   i, j] / args.data_stride ** 2

                yr0 = y0 - dy * i * patch_H/args.data_stride
                yrn = yr0 - dy * patch_H
                xr0 = x4000 + dx *j * patch_W/args.data_stride
                xrn = xr0 + dx * patch_W
                rectangle = box(xr0, yrn, xrn, yr0)
                rectangle_gdf = gpd.GeoDataFrame(geometry=[rectangle])
                ints_area = rectangle_gdf.intersection(mask_polyg).area

                is_within_mask = mask_polyg.geometry.apply(lambda poly: rectangle.within(poly)).any()
                intersection_areas = []

                # for p in mask_polyg.geometry:
                #     if rectangle.intersects(p):
                #         intersection = rectangle.intersection(p)
                #         intersection_areas.append(intersection.area)
                # intersection_area = sum(intersection_areas) if len(intersection_areas) > 0 else 0
                # relative_intersection = intersection_area / rectangle.area


                # fig, ax = plt.subplots()
                #
                # # Plot both GeoDataFrames on the same plot
                # mask_polyg.plot(ax=ax, edgecolor='blue', facecolor='none', label='Polygon 1')
                # rectangle_gdf.plot(ax=ax, edgecolor='red', facecolor='none', label='Polygon 2')
                #
                #
                # ax.set_title(str(relative_intersection))
                #
                #
                # plt.show()
                # plt.close()
                #if relative_intersection > 0.5:
                if is_within_mask:

                    # if  np.any(data[i,j]>5):
                    image = torch.tensor(data[i,j]).unsqueeze(0).unsqueeze(1).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    pred = net(image)
                    pred = (F.sigmoid(pred) > 0.5).float()
                    pred = pred.squeeze(1).squeeze(0).cpu().detach().numpy()


                    reconstructed_pred[i * patch_H // args.data_stride :i* patch_H // args.data_stride + patch_H , j * patch_W // args.data_stride : j * patch_W // args.data_stride + patch_W] += pred/args.data_stride**2


                # fig, (ax1,ax2, ax3) = plt.subplots(1, 3)
                # ax1.imshow(data[i,j])
                # ax2.imshow(mask[i,j])
                # ax3.imshow(pred)
                # plt.show()
        reconstructed_pred = np.where(reconstructed_pred > args.recon_th, 1, 0).astype(np.float32)
        transform = Affine.identity()  # Create an identity transform
        polygons = []
        for geom, val in rasterio.features.shapes(reconstructed_pred, transform=transform):
            if val ==1:
                polygons.append(shape(geom))

        polygons_gpd = gpd.GeoDataFrame(geometry=polygons)

        prefix = output_path  + intf
        out_polyg_path = prefix + '_predicted_polyogns.shp'
        polygons_gpd.to_file(out_polyg_path)


        np.save(prefix +'_pred', reconstructed_pred)
        np.save(prefix +'_image', reconstructed_intf)
        np.save(prefix +'_gt', reconstructed_mask)


        if args.plot:
            fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))

            # ax1.imshow(full_intf_data)
            # ax1.set_title('Orig Image')

            ax1.imshow(reconstructed_intf)
            ax1.set_title('Reconstructed Image')
            ax2.imshow(reconstructed_mask)
            ax2.set_title('Reconstructed True mask')
            ax3.imshow(reconstructed_pred)
            ax3.set_title('pred mask')
            def on_xlims_change(axes):
                for ax in (ax1, ax2, ax3):
                    if ax != axes:
                        ax.set_xlim(axes.get_xlim())


            def on_ylims_change(axes):
                for ax in (ax1, ax2, ax3):
                    if ax != axes:
                        ax.set_ylim(axes.get_ylim())


            # Connect the events
            ax1.callbacks.connect('xlim_changed', on_xlims_change)
            ax1.callbacks.connect('ylim_changed', on_ylims_change)

            # Show the plot
            plt.show()





