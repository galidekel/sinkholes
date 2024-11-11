import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import argparse
from pathlib import Path
import os
from train_sinkholes_unet import str2bool

logging.basicConfig(level=logging.INFO)


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
    if mask is None:
       return np.array(data_patches)
    else:
        return np.array(data_patches),np.array(mask_patches), np.array(data_patches_nonz), np.array(mask_patches_nonz)
def get_args():
    parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')
    parser.add_argument('--by_list',  type=str, default=None, help='From an input list')
    parser.add_argument('--input_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/', help='full unw files directory for input')
    parser.add_argument('--output_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='patches output')
    parser.add_argument('--gt_polygon_file_path',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/sub_20231001.shp', help='')
    parser.add_argument('--plot_data',  type=bool, default=False)
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--strides_per_patch',type=int, default=2, help='strides per patch - 2 means half a window stride, 4 means quarter a window stride etc')
    parser.add_argument('--intf_22_23',  type=str, default='False')

    parser.add_argument('--days_diff',  type=int, default=11)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    req_days_diff = args.days_diff
    args.intf_22_23 = str2bool(args.intf_22_23)

    gdf = gpd.read_file(args.gt_polygon_file_path)
    patch_size = tuple(args.patch_size)
    patch_H, patch_W = patch_size
    data_output_dir = args.output_dir + 'data_patches_H' + str(patch_H) + '_W' + str(patch_W) + '_strpp'+str(args.strides_per_patch) + ('_'+str(req_days_diff)+'days') + ('_22_23' if args.intf_22_23 else '')
    mask_output_dir = args.output_dir + 'mask_patches_H' + str(patch_H) + '_W' + str(patch_W) + '_strpp'+str(args.strides_per_patch) + ('_'+str(req_days_diff)+'days') +  ('_22_23' if args.intf_22_23 else '')


    for item in [data_output_dir, mask_output_dir]:
        if not os.path.exists(item):
            # If it doesn't exist, create it
            os.makedirs(item)
        else:
            logging.info('Note! These patches arleadt exist! If you want to recreate them - remove the existing dir first')
            #sys.exit()
    if args.by_list is None:
        input_files_paths = list(Path(args.input_dir).glob('*.unw'))
    else:
        input_files = []
        input_file_paths = []
        input_intfs = args.by_list.split(',')
        for filename in os.listdir(args.input_dir):
            if filename.endswith('.unw') and any(input_intf[:8] ==  filename[9:17] and input_intf[9:]==filename[25:33]for input_intf in input_intfs):
                input_files.append(filename)
                input_file_paths = [Path(args.input_dir) / file for file in input_files]

    for item in input_file_paths:
        gfile_name = item.name
        intfrgrm_name = gfile_name.split('.')[0][9:17]+gfile_name.split('.')[0][24:33]
        start_date = intfrgrm_name[0:4]+'-'+intfrgrm_name[4:6]+'-'+intfrgrm_name[6:8]
        end_date = intfrgrm_name[9:13]+'-'+intfrgrm_name[13:15]+'-'+intfrgrm_name[15:17]

        start_datetime = datetime.strptime(start_date,"%Y-%m-%d")
        end_datetime = datetime.strptime(end_date,"%Y-%m-%d")
        days_diff = (end_datetime - start_datetime).days
        if args.intf_22_23:
            if days_diff != req_days_diff or end_datetime.year < 2022:
                continue
        elif days_diff != req_days_diff or start_datetime.year >2021:
            continue

        with open(args.input_dir+'/'+gfile_name + '.ers') as f:
               for line in f:
                  if 'NrOfLines' in line:
                     NLINES = int(line.strip().split()[-1])
                  elif 'NrOfCellsPerLine' in line:
                    NCELLS = int(line.strip().split()[-1])
                  if 'ByteOrder' in line:
                     byte_order = line.strip().split()[-1]
                  if 'Eastings' in line:
                     x0 = float(line.strip().split()[-1])
                  if 'Northings' in line:
                     y0 = float(line.strip().split()[-1])

        data = np.fromfile(args.input_dir+'/'+gfile_name, dtype=np.float32, count=-1, sep='', offset=0).reshape(NLINES, NCELLS)
        if byte_order == 'MSBFirst':
            data = data.byteswap().newbyteorder('<')


        subset_polygs = gdf[(gdf['start_date']==start_date) & (gdf['end_date']==end_date)]
        if subset_polygs.empty:
            logging.info('no GT polygons for interferogram: ' + intfrgrm_name)
            continue
        transform = rasterio.transform.from_origin(x0, y0, 0.00002777, 0.00002777)  # Adjust resolution and (North,West) origin to the interferogram
        height, width = NLINES,NCELLS  # Adjust dimension to match the interferogram
        profile = {
            'driver': 'GTiff',
            'count': 1,
            'height': height,
            'width': width,
            'dtype': 'uint8',
            'crs': subset_polygs.crs,
            'transform': transform
        }

        #Create the mask using rasterio's geometry_mask
        mask = geometry_mask(subset_polygs.geometry, transform=transform, invert=True, out_shape=(height, width)).astype(np.uint8)
        extent = [x0, x0 + 0.00002777 * (NCELLS), y0 - 0.00002777 * (NLINES), y0]
        logging.info('long lat range: {}'.format(extent))

        if args.plot_data:
            plt.rcParams['backend'] = 'Qt5Agg'

            plt.imshow(data, extent=extent, cmap='Greys')
            plt.title('Interferogram '+ start_date + ' - ' + end_date)
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.show()
            fig, ax = plt.subplots(1,2,figsize=(10,4), sharey=True, sharex=True,gridspec_kw={'width_ratios': [1, 1]})

            subset_polygs.plot(ax=ax[0], edgecolor='blue', facecolor='none')
            ax[0].set_title('subset polygons')
            mask_imag = ax[1].imshow(mask,cmap='gray',extent=extent)
            ax[1].set_title('subset polygons mask')
            fig.colorbar(mask_imag, ax=ax[1], shrink=0.8)
            plt.show()
            plt.rcParams['backend'] = 'Qt5Agg'

            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True,
                                   gridspec_kw={'width_ratios': [1, 1]})

            c0 = ax[0].imshow(data, extent=extent, cmap='gray')
            ax[0].set_title(f'Interferogram'+intfrgrm_name, fontsize=12)
            ax[0].set_xlabel(f'Longitude (deg)')
            ax[0].set_ylabel(f'Latitude (deg)')

            ax[1].imshow(mask, extent=extent, cmap='gray', vmin=0)
            colorbar = fig.colorbar(c0, ax=ax[0], shrink=0.5)
            colorbar.ax.tick_params(labelsize=7)

            for i in range(len(subset_polygs)):
                coords_e = subset_polygs.iloc[i]['geometry'].exterior.coords
                coords_list = list(coords_e)
                x, y = zip(*coords_list)
                ax[0].plot(x, y)
                ax[1].plot(x, y)
            plt.tight_layout()
            plt.show()

        # Write the mask to a raster file
        # with rasterio.open("subset_mask.tif", 'w', **profile) as dst:
        #     dst.write(mask, 1)
        # with rasterio.open("subset_mask.tif") as src:
        #     # Read the mask data into a NumPy array
        #     mask_array = src.read(1)


        data_patches,mask_patches,data_patches_nonz,mask_patches_nonz = patchify_for_train_and_test(data, patch_size, stride= (patch_size[0] // args.strides_per_patch, patch_size[1] // args.strides_per_patch), mask_array=mask)
        counter_nonz = data_patches_nonz.shape[0]
        logging.info('intrfrgrm {}: number of non-zero patches: '.format(intfrgrm_name)+str(counter_nonz))

        ext =  intfrgrm_name +'_H'+str(patch_H)+'_W'+str(patch_W)+'_strpp'+str(args.strides_per_patch)+'.npy'
        out_file_names = ['data_patches_','mask_patches_','data_patches_nonz_','mask_patches_nonz_']
        for i,array in enumerate([data_patches,mask_patches,data_patches_nonz,mask_patches_nonz]):
            if 'mask' in out_file_names[i]:
                output_dir = mask_output_dir
            else:
                output_dir = data_output_dir
            out_file_name = out_file_names[i] + ext
            np.save(output_dir+'/'+ out_file_name,array)


        #test
            test_array = np.load(output_dir+'/'+ out_file_name)
            logging.info('shpae of file saved: {}'.format(test_array.shape))


        if args.plot_data:
            mask_patches = np.load(mask_output_dir + '/' + 'mask_patches_nonz_' + ext)
            data_patches = np.load(data_output_dir + '/' + 'data_patches_nonz_' + ext)
            for i in range(0,len(data_patches)-8,8):
               fig, axes = plt.subplots(4, 4, figsize=(10, 10),sharex=True, sharey=True)
               axes = axes.flatten()
               full = []
               for j in range(8):
                    axes[2*j].imshow(data_patches[i+j],cmap='gray')
                    axes[2*j+1].imshow(mask_patches[i+j],cmap='gray')
                    axes[2*j+1].set_xlim(axes[2*j].get_xlim())
                    axes[2*j+1].set_ylim(axes[2*j].get_ylim())
                    axes[2*j].tick_params(axis='x', labelsize=7)
                    axes[2*j].tick_params(axis='y', labelsize=7)
                    axes[2 * j+1].tick_params(axis='x', labelsize=7)
                    axes[2 * j+1].tick_params(axis='y', labelsize=7)
                    axes[2*j].set_title('Patch {}'.format(i+j),fontsize=7)
                    full.append(mask_patches[i+j])
               full = np.array(full)
               if np.any(full>0):
                  plt.show()
               else:
                 plt.close()