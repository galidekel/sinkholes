import numpy as np
from matplotlib import pyplot as plt
from rasterio.features import shapes
from affine import Affine
import rasterio
from rasterio.features import rasterize

import rasterio.features
from shapely.geometry import shape, Polygon, mapping

from shapely.geometry import Polygon, LineString

import argparse
from os import listdir, makedirs, path

def get_args():

    parser = argparse.ArgumentParser(description='clean edges from mask patches')
    parser.add_argument('--patches_path', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/' , help='patch H, patch W')

    parser.add_argument('--patch_size', nargs='+', type=int, default=[200, 100], help='patch H, patch W')
    parser.add_argument('--eleven_days_diff', type=str, default='True')
    parser.add_argument('--strides_per_patch', type=int, default=2)
    return parser.parse_args()
if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    args = get_args()
    H , W = args.patch_size[0], args.patch_size[1]
    patch_path = args.patches_path + 'mask_patches'+ '_H' + str(H) + '_W' + str(W) +'_strpp' + str(args.strides_per_patch) + '_11days' '/'
    makedirs(patch_path+'cleaned/', exist_ok=True)

    files = [item for item in listdir(patch_path) if path.isfile(path.join(patch_path, item))]


    for n,file in enumerate(files):
        patch_boundary = LineString([(0, 0), (W, 0), (W, H), (0, H), (0, 0)])

        patch_data = np.load(patch_path+file).astype(np.float32)
        orig_shape = patch_data.shape
        if len(orig_shape) == 4:
            patch_data = np.reshape(patch_data,(orig_shape[0]* orig_shape[1],orig_shape[2],orig_shape[3]))
        transform = Affine.identity()  # Create an identity transform
        threshold = 20
        topolg_transform = Affine.identity()
        cleaned_data = np.zeros_like(patch_data)
        for i in range(patch_data.shape[0]):
            if 'nonz' not in file and not np.any(patch_data[i]):
                continue
            # plt.imshow(patch_data[i,:,:], cmap='grey')
            # plt.show()
            shapes = rasterio.features.shapes(patch_data[i], patch_data[i] > 0)
            polygons = [shape(geom) for geom, value in shapes if value > 0]
            # for p in polygons:
            #     coords_e = p.exterior.coords
            #     coords_list = list(coords_e)
            #     x, y = zip(*coords_list)
            #     plt.gca().set_aspect('equal', adjustable='box')
            #
            #     plt.plot(x,y)
            #     plt.xlim(0,100)
            #     plt.ylim(200,0)
            #
            #     plt.show()
            # plt.show()

            removed = False
            for item in polygons:
                min_x = min([point[0] for point in item .exterior.coords])
                max_x = max([point[0] for point in item.exterior.coords])
                min_y = min([point[1] for point in item .exterior.coords])
                max_y = max([point[1] for point in item.exterior.coords])


                if item.area < threshold and item.touches(patch_boundary) or (max_x-min_x<10 and (max_x==W or min_x==1)) or (max_y-min_y<10 and (max_y==H or min_y==1)):
                    polygons.remove(item)
                    removed = True

            transform = Affine.identity()
            # Rasterize the polygons
            if len(polygons)>0:
                cleaned_data[i] = rasterio.features.rasterize(
                    [(mapping(polygon), 1) for polygon in polygons],
                    out_shape=(H, W),
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
            )
            else:
                cleaned_data[i] = np.zeros((H, W), dtype=np.uint8)
            if removed and False:
                fig, [ax1,ax2] = plt.subplots(1,2, figsize=(10,10))
                ax1.imshow(patch_data[i] , cmap='gray')
                ax1.set_title('original mask')
                ax2.imshow(cleaned_data[i], cmap='gray')
                ax2.set_title('cleaned mask')
                plt.show()
        print('finished cleaning inft'+str(n))
        if len(orig_shape) ==4:
            cleaned_data = np.reshape(cleaned_data,(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[3]))
        np.save(patch_path+'cleaned/'+file.split('.')[0] +'_cleaned.npy', cleaned_data)
        reloaded = np.load(patch_path+'cleaned/'+file.split('.')[0] +'_cleaned.npy')
        print('shaped of saved cleaned data is' + str(np.shape(reloaded)))
        assert reloaded.shape == orig_shape, 'Not as original!'