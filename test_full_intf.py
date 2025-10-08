#!/usr/bin/env python3
import sys
import pickle
import os
import json
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

import torch
import torch.nn.functional as F  # only if you need F.*; otherwise torch.sigmoid used below

from unet import *
from polygs import *
from get_intf_info import *
from attn_unet import *

def str2bool(arg):
    return arg.lower() == 'true'

def reconstruct_intf_prediction(data, intf_coords, net, patch_size, stride, rth,
                                mask=None, add_lidar_mask=True, plot=False):
    """
    Args:
      data: (C, ny, nx, H, W) or (ny, nx, H, W) where channel 0 is current and 1..C-1 are prevs.
    Returns:
      If mask is None:
        reconstructed_intf, reconstructed_pred, reconstructed_prevs
      Else:
        reconstructed_intf, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs
    """
    x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask, num_nonz_p, bo, frame = intf_coords
    x4000 = x0 + 3000 * dx  # hack!! delete
    patch_H, patch_W = patch_size

    # Normalize shape views
    if data.ndim == 5:                      # (C, ny, nx, H, W)
        C, ny, nx, H, W = data.shape
        data_stack = data.astype(np.float32, copy=False)
        curr_data  = data_stack[0]          # (ny, nx, H, W)
    else:                                   # (ny, nx, H, W)
        ny, nx, H, W = data.shape
        C = 1
        data_stack = data[None].astype(np.float32, copy=False)  # (1, ny, nx, H, W)
        curr_data  = data_stack[0]

    out_h = ny * patch_H // stride + patch_H * (1 - 1 // stride)
    out_w = nx * patch_W // stride + patch_W * (1 - 1 // stride)

    # Reconstruct ALL channels: current + prevs
    reconstructed_intf_all = np.zeros((C, out_h, out_w), dtype=np.float32)

    reconstructed_pred = np.zeros((out_h, out_w), dtype=np.float32)
    if mask is not None:
        reconstructed_mask = np.zeros((out_h, out_w), dtype=np.float32)

    # Optional LiDAR mask
    if add_lidar_mask:
        lidar_mask_df = gpd.read_file('lidar_mask_polygs.shp')
        mask_polyg = lidar_mask_df[lidar_mask_df['source'] == intf_lidar_mask]
        rasterized_polygon = rasterize(
            [(g, 1) for g in mask_polyg['geometry'].tolist()],
            out_shape=(out_h, out_w),
            transform=rasterio.transform.from_origin(x4000, y0, dx, dy),
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

    # Main loop
    for i in range(ny):
        print(i)
        for j in range(nx):
            y0o = i * (patch_H // stride); y1o = y0o + patch_H
            x0o = j * (patch_W // stride); x1o = x0o + patch_W

            # Accumulate ALL channels (current + prevs)
            # (keeps your /stride**2 averaging scheme)
            for c in range(C):
                reconstructed_intf_all[c, y0o:y1o, x0o:x1o] += data_stack[c, i, j] / (stride ** 2)

            if mask is not None:
                reconstructed_mask[y0o:y1o, x0o:x1o] += mask[i, j] / (stride ** 2)

            is_within_mask = True
            if add_lidar_mask:
                is_within_mask = np.all(rasterized_polygon[y0o:y1o, x0o:x1o])

            if is_within_mask:
                # Build (1, C, H, W) for net and predict
                x_np = data_stack[:, i, j]  # (C, H, W), already float32
                image = torch.from_numpy(x_np[None]).to(device=device, memory_format=torch.channels_last)
                with torch.no_grad():
                    pred = (torch.sigmoid(net(image)) > 0.5).float().squeeze().cpu().numpy()  # (H, W)

                reconstructed_pred[y0o:y1o, x0o:x1o] += pred / (stride ** 2)

    reconstructed_pred_th = (reconstructed_pred > rth).astype(np.float32)

    # Split current and prev outputs
    reconstructed_intf = reconstructed_intf_all[0]
    reconstructed_prevs = reconstructed_intf_all[1:] if C > 1 else None

    if mask is None:
        return reconstructed_intf, reconstructed_pred, reconstructed_prevs
    else:
        return reconstructed_intf_all, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs


def get_pred_args():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')
    parser.add_argument('--full_intf_dir',  type=str, default='./', help='full interferogram path')
    parser.add_argument('--input_patch_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='patches inputs')
    parser.add_argument('--plot_data',  type=bool, default=False)
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--days_diff', type=int, default=11)

    parser.add_argument('--intf_source', type=str, default = 'intf_list', choices=['intf_list', 'test_dataset','preset','all'])
    parser.add_argument('--intf_list', type=str, default = None, help='a list of intf ids divided by comma')
    parser.add_argument('--test_dataset', type=str, default = None, help='path to teset_dastset')
    parser.add_argument('--valset_from_partition', type=str, default=None, help='val set from a partition_File')

    parser.add_argument('--model', '-m', default='checkpoint_epoch28.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

    parser.add_argument('--data_stride', type=int, default=2)
    parser.add_argument('--recon_th', type=float, default=0.25)

    parser.add_argument('--job_name', type=str, default='job', help='unique job name')
    parser.add_argument('--add_lidar_mask', type=str,default='True')

    parser.add_argument('--plot' ,action='store_true')
    parser.add_argument('--attn_unet',action='store_true')
    parser.add_argument('--years_22_23',action='store_true')
    parser.add_argument('--unioned_mask',action='store_true')

    parser.add_argument('--k_prevs', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    logging.basicConfig(level=logging.INFO)
    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)
    now = datetime.now().strftime("%m_%d_%Hh%M")

    args = get_pred_args()
    args.add_lidar_mask = str2bool(args.add_lidar_mask)
    if not is_running_locally:
        args.plot = False

    job_name = args.job_name +'_'+ now
    model_name = args.model.split('.')[0]
    output_path = f'pred_outputs2/{model_name}/{job_name}/'
    output_polyg_path = output_path + 'polygs/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_polyg_path, exist_ok=True)
    log_file = output_path + args.job_name + '_' + now + '.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)

    logging.info('Running job {} with model {}.pth and args: {}'.format(job_name, model_name, args))

    if (args.intf_source == 'intf_list' and args.intf_list is None) or \
       (args.intf_source == 'test_dataset' and args.test_dataset is None) or \
       (args.intf_source == 'preset' and args.valset_from_partition is None):
        logging.info('you chose to take intfs from '+args.intf_source + ' but it is None. exiting.')
        sys.exit()

    if args.intf_source == 'intf_list':
        intf_list = args.intf_list.split(',')
    elif args.intf_source == 'preset':
        with open(args.valset_from_partition, 'r') as file:
            loaded_data = json.load(file)
        intf_list = loaded_data['val']
        logging.info('taking test intfs from partition {}. intf list: {}'.format(args.valset_from_partition, intf_list))
    else:
        with open('./test_data/'+args.test_dataset, 'rb') as file:
            test_data = pickle.load(file)
        intf_list = test_data.ids
        logging.info('taking test intfs from test dataset {}. intf list: {}'.format(args.test_dataset, intf_list))

    # Model
    net = UNet(n_channels=args.k_prevs + 1, n_classes=1, bilinear=False)
    if args.attn_unet:
        net = AttentionUNet(n_channels=1, n_classes=1, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    state_dict = torch.load('./models/'+args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.eval()
    logging.info('Model loaded!')

    # Data dirs
    patch_H, patch_W = args.patch_size
    data_dir = args.input_patch_dir + \
        f'data_patches_H{patch_H}_W{patch_W}_strpp{args.data_stride}' + \
        (f'_{args.days_diff}days') + ('_22_23' if args.years_22_23 else '') + '_Aligned'
    mask_dir = args.input_patch_dir + \
        f'mask_patches_H{patch_H}_W{patch_W}_strpp{args.data_stride}' + \
        (f'_{args.days_diff}days') + ('_22_23' if args.years_22_23 else '') + '_Aligned'

    # If using prevs, precompute the sequences
    if args.k_prevs > 0:
        with open('intf_coord.json', "r") as f:
            intf_info = json.load(f)
        prev_dict, updated_intf_list = find_11day_sequences(intf_info, k_prev=args.k_prevs,
                                                            restrict_to=intf_list)
        intf_list = updated_intf_list
    else:
        prev_dict = None

    for intf in intf_list:
        ic = get_intf_coords(intf)
        if args.k_prevs > 0:
            if ic[11] == 'North':
                intfs_coords = (35.3, 31.79) + ic[2:]
            else:
                intfs_coords = (35.25, 31.44) + ic[2:]

        data_file_name = f'data_patches_{intf}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy'
        mask_file_name = f'mask_patches_{intf}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy'
        data_path = os.path.join(data_dir, data_file_name)
        mask_path = os.path.join(mask_dir, mask_file_name)

        # Load current (normalized or radians; we’ll normalize-if-needed below)
        cur = np.load(data_path).astype(np.float32)  # [ny, nx, H, W]

        # Stack prevs if requested
        if args.k_prevs > 0:
            prev_ids = prev_dict[intf]['prevs'][:args.k_prevs][::-1]  # newest-first
            prevs = [
                np.load(os.path.join(data_dir,
                        f'data_patches_{pid}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy')
                       ).astype(np.float32)
                for pid in prev_ids
            ]
            pa = [cur] + prevs  # list of (ny, nx, H, W)

            # Crop trailing edges to common overlap
            ny = min(p.shape[0] for p in pa)
            nx = min(p.shape[1] for p in pa)
            H, W = pa[0].shape[2], pa[0].shape[3]

            # Normalize each array only if not already in [0,1]
            tol = 1e-3
            normed = []
            for p in pa:
                pc = p[:ny, :nx, :H, :W].astype(np.float32, copy=False)
                mn = float(np.nanmin(pc)); mx = float(np.nanmax(pc))
                if (mn >= -tol) and (mx <= 1.0 + tol):
                    q = pc.copy()  # only copy when we may modify
                    zmask = (q == 0.0)  # only sharp zeros
                    if zmask.any():
                        q[zmask] = 0.5
                    normed.append(q)
                else:
                    normed.append((pc + np.pi) / (2 * np.pi))
            data = np.stack(normed, axis=0).astype(np.float32)  # (C, ny, nx, H, W)
        else:
            # Single-channel: normalize if needed
            mn = float(np.nanmin(cur)); mx = float(np.nanmax(cur))
            if (mn >= -tol) and (mx <= 1.0 + tol):
                q = cur.copy()  # copy only if we'll modify
                zmask = (q == 0.0)  # only sharp zeros
                if zmask.any():
                    q[zmask] = 0.5
                data = q[None]
            else:
                data = ((cur + np.pi) / (2 * np.pi))[None]
        mask_cur = np.load(mask_path).astype(np.float32)  # (ny?, nx?, H, W)

        if args.k_prevs > 0 and args.unioned_mask:
            # prev mask files
            prev_mask_paths = [
                os.path.join(mask_dir, f'mask_patches_{pid}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy')
                for pid in prev_ids
            ]
            prev_masks = [np.load(p).astype(np.float32) for p in prev_mask_paths]

            # crop all masks to match the (ny, nx, H, W) we used for data
            ma = [mask_cur] + prev_masks
            ma = [m[:ny, :nx, :H, :W] for m in ma]

            # union over time: any positive → 1
            mask = (np.stack(ma, axis=0) > 0).any(axis=0).astype(np.float32)  # (ny, nx, H, W)
        else:
            # no union: just make sure shape matches data if we cropped
            mask = mask_cur[:ny, :nx, :H, :W] if args.k_prevs > 0 else mask_cur

        logging.info(f"{mask_file_name} shape: {mask.shape}")

        # Reconstruct (now returns prev reconstructions too)
        reconstructed_intf, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs = \
            reconstruct_intf_prediction(data, intfs_coords, net, (patch_H, patch_W),
                                        args.data_stride, args.recon_th, mask,
                                        add_lidar_mask=args.add_lidar_mask)

        x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask, num_nz, bo, frame = intfs_coords

        # Convert mask array to polygons with lat/lon coordinates
        polygons = plg_indx2longlat(mask_array_to_polygons(reconstructed_pred_th), intfs_coords)
        list_of_polygons = polygons.geometry.tolist()

        prefix = os.path.join(output_path, intf)
        out_polyg_path = os.path.join(output_polyg_path, f'{intf}_predicted_polyogns.shp')
        polygons.to_file(out_polyg_path)

        np.save(prefix + '_pred_th', reconstructed_pred_th)
        np.save(prefix + '_pred', reconstructed_pred)
        np.save(prefix + '_image', reconstructed_intf)
        np.save(prefix + '_gt', reconstructed_mask)
        # Optionally save prev reconstructions
        if reconstructed_prevs is not None and reconstructed_prevs.shape[0] > 0:
            np.save(prefix + '_image_prevs', reconstructed_prevs)

        if args.plot:
            k_prev = 0 if reconstructed_prevs is None else reconstructed_prevs.shape[0]
            # current + prevs + (true mask if present) + pred mask
            has_mask = reconstructed_mask is not None
            ncols = 1 + k_prev + (1 if has_mask else 0) + 1

            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5), sharex=True, sharey=True)
            axes = np.atleast_1d(axes)

            x4000p = x0 + 3000*dx
            extent = [x4000p, x4000p + dx * reconstructed_intf.shape[1],
                      y0 - dy * reconstructed_intf.shape[0], y0]

            col = 0
            axes[col].imshow(reconstructed_intf, extent=extent, cmap='jet')
            axes[col].set_title('Current');
            col += 1

            if reconstructed_prevs is not None:
                for p in range(k_prev):
                    axes[col].imshow(reconstructed_prevs[p], extent=extent, cmap='jet')
                    axes[col].set_title(f'Prev t-{p + 1}')
                    col += 1

            if has_mask:
                axes[col].imshow(reconstructed_mask, extent=extent);
                axes[col].set_title('True mask');
                col += 1

            axes[col].imshow(reconstructed_pred_th, extent=extent);
            axes[col].set_title('Pred mask')

            for poly in list_of_polygons:
                xg, yg = poly.exterior.xy
                axes[0].plot(xg, yg, color='k', linewidth=1)

            plt.tight_layout()
            plt.show()


