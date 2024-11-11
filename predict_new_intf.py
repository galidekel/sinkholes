import argparse
import os
from pathlib import Path
import numpy as np
from prepare_intrfrgrm_pathches import patchify
from unet import *
import torch
import logging
from get_intf_info import *
from test_full_intf import reconstruct_intf
from polygs import *
import matplotlib.pyplot as plt
def get_args():
    parser = argparse.ArgumentParser(description='Predict nes intfs')
    parser.add_argument('--intfs_dir',  type=str, default='./', help='full interferogram path')
    parser.add_argument('--intfs_list', type=str, default='a list of intfs in format YYYYMMDD_YYYYMMDD seperated by comma')
    parser.add_argument('--model_dir', type=str,default='./models/')
    parser.add_argument('--model_file', type=str)

    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--stride',type = int, default = 2)
    parser.add_argument('--plot_data',  type=bool, default=False)
    parser.add_argument('--output_polygon_dir', type=str)
    parser.add_argument('--rth',type = float, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
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
        with open(p+ '.ers') as f:
            for line in f:
                if 'ByteOrder' in line:
                    byte_order = line.strip().split()[-1]
        intf = intfs_list[i]
        intfs_coords = get_intf_coords(intf)
        x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nz = intfs_coords

        data = np.fromfile(p, dtype=np.float32, count=-1, sep='', offset=0).reshape(nlines, ncells)
        if byte_order == 'MSBFirst':
            data = data.byteswap().newbyteorder('<')
        patchified_intf = patchify(data, args.patch_size, args.stride, mask_array=None, nonz_pathces=False)

        reconstructed_intf,reconstructed_pred = reconstruct_intf(data,intfs_coords,net,args.stride,args.rth)
        polygons = plg_indx2longlat(mask_array_to_polygons(reconstructed_pred),intfs_coords)

        if not os.path.exists(args.output_polygs_dir):
            os.makedirs(args.output_polygs_dir)
        out_polyg_f = args.output_polygs_dir + intf +'_predicted_polyogns.shp'
        polygons.to_file(out_polyg_f)

        x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nz = intfs_coords

        extent = [x4000, x4000 + dx * reconstructed_intf.shape[1], y0 - dy * reconstructed_intf.shape[0], y0]

        plt.imshow(reconstructed_intf, extent=extent)
        list_of_polygons = polygons.geometry.tolist()

        for p in list_of_polygons:
            x, y = p.exterior.xy
            plt.plot(x, y)


        plt.show()




