import  argparse
import torch
from unet import *
import logging
import os
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from get_intf_info import *


def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg

def get_args():
    parser = argparse.ArgumentParser(description='run model on full intf and add fpz patches to retraining')
    parser.add_argument('--patch_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[200, 100], help='patch H, patch W')
    parser.add_argument('--stride', type=int, default=2, help='train on patchs with given strides per window')
    parser.add_argument('--train_on_11d_diff', type = str, default='True')
    parser.add_argument('--test_list_to_exclude', type = str, help='test intfs to exclude from retraining seperated by comma')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--intf_dict_path', type=str, default='./intf_coord.json', help='path to interferograms coord dict')
    parser.add_argument('--nonz_th',  nargs = '+', type = int, default=[350,150], help='north, south')


    return  parser.parse_args()
if __name__ == '__main__':
    args = get_args()
    args.train_on_11d_diff = str2bool(args.train_on_11d_diff)
    if args.test_list_to_exclude is not None:
        test_list = args.test_list_to_exclude.split(',')
    else:
        test_list = []
    H,W = args.patch_size
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model_file}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    state_dict = torch.load('./models/'+args.model_file, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.eval()
    logging.info('Model loaded!')
    image_dir = args.patch_dir + 'data_patches_H' + str(H) + '_W' + str(W) + '_strpp' + str(args.stride) + (
        '_11days/' if args.train_on_11d_diff else '_all/')
    mask_dir = args.patch_dir + 'mask_patches_H' + str(H) + '_W' + str(W) + '_strpp' + str(args.stride) + (
        '_11days/' if args.train_on_11d_diff else '_all/')
    output_image_patch_dir = image_dir +'additional_fp_patches/'
    output_mask_patch_dir =  mask_dir + 'additional_fp_patches/'
    os.makedirs(output_image_patch_dir, exist_ok=True)
    os.makedirs(output_mask_patch_dir, exist_ok=True)
    intf_list = [item[13:30] for item in os.listdir(image_dir) if 'nonz' not in item and os.path.isfile(image_dir+item)]
    n1 = len(intf_list)
    logging.info('Original list has {} nonz'.format(n1))

    with open(args.intf_dict_path, 'r') as json_file:
        coord_dict = json.load(json_file)
    nonz_th_north, nonz_th_south = tuple(args.nonz_th)[0], tuple(args.nonz_th)[1]
    filtered_list = []
    for intf in intf_list:
        if (coord_dict[intf]['north'] > 31.5 and coord_dict[intf]['nonz_num'] > nonz_th_north) or (
                coord_dict[intf]['north'] < 31.5 and coord_dict[intf]['nonz_num'] > nonz_th_south):
            filtered_list.append(intf)
    intf_list = filtered_list
    n2 = len(intf_list)
    logging.info('filtered list has {} nonz'.format(n2))

    logging.info('filtered by nonz threshold: removed {} intfs'.format(n1 - n2))
    assert all(item in intf_list for item in test_list), f'Test list provided is NOT included in the current data set check again! '

    intf_list = list(set(intf_list) - set(test_list))
    logging.info('removed {} intfs \nintf list is: {}'.format(len(test_list),intf_list))

    for intf in intf_list:
        x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask, num_nonz_p, bo = get_intf_coords(intf)

        data_file_name = 'data_patches_' + intf + '_H' + str(H) + '_W' + str(W)+'_strpp{}'.format(args.stride) +'.npy'
        mask_file_name = 'mask_patches_' + intf + '_H' + str(H) + '_W' + str(W)+'_strpp{}'.format(args.stride) +'.npy'
        data_path = image_dir + data_file_name
        mask_path = mask_dir + mask_file_name
        data = np.load(data_path)
        mask = np.load(mask_path)
        assert data.shape == mask.shape, 'data shape and mask shape do not match!!!'
        data = (data + np.pi) / (2 * np.pi)
        add_lidar_mask = True
        if add_lidar_mask:
            lidar_mask_df = gpd.read_file('lidar_mask_polygs.shp')
            mask_polyg = lidar_mask_df[lidar_mask_df['source'] == intf_lidar_mask]
            height, width =(data.shape[0] * H // args.stride + H * (1 - 1 // args.stride),
                                   data.shape[1] * W // args.stride + W * (1 - 1 // args.stride))
            transform = rasterio.transform.from_origin(x4000, y0, dx, dy)  # Example transform

            # Create an empty raster array
            raster = np.zeros((height, width), dtype=np.uint8)
            geometries = mask_polyg['geometry'].tolist()
            # Rasterize the polygon
            rasterized_polygon = rasterize(
                [(g, 1) for g in geometries],  # List of (geometry, value) tuples
                out_shape=raster.shape,
                transform=transform,
                fill=0,
                all_touched=True,  # If True, all pixels touched by geometries will be burned in
                dtype=raster.dtype
            )
        fpz_list = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                is_within_mask = True

                if add_lidar_mask:
                    is_within_mask = np.all(rasterized_polygon[
                                            i * H // args.stride:i * H // args.stride + H,
                                            j * W // args.stride: j * W // args.stride + W])

                if is_within_mask and  np.all(mask[i,j]==0):


                    image = torch.tensor(data[i, j]).unsqueeze(0).unsqueeze(1).to(device=device,
                                                                              dtype=torch.float32,
                                                                              memory_format=torch.channels_last)
                    pred = net(image)
                    pred = (F.sigmoid(pred) > 0.5).float()
                    pred = pred.squeeze(1).squeeze(0).cpu().detach().numpy()
                    if np.any(pred > 0):
                        print(i, j)

                        # fig,axes = plt.subplots(1,2)
                        # axes[0].imshow(pred, cmap='gray')
                        # axes[1].imshow(mask[i,j], cmap='gray')
                        # plt.show()
                        data_patch_name = 'data_patches_fp_'+str(intf)
                        fpz_list.append(data[i,j])
                        data_array = np.array(fpz_list)
        np.save(output_image_patch_dir+ data_patch_name, data_array)






