import sys
import pickle
import rasterio.features
from unet import *
from polygs import *
import os
import rasterio
from rasterio.features import rasterize
import logging
from datetime import datetime
from get_intf_info import *
from attn_unet import *

def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg

def reconstruct_intf_prediction(data, intf_coords, net,patch_size, stride, rth, mask = None, add_lidar_mask = True, plot = False):
    x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nonz_p,bo = intf_coords
    patch_H,patch_W = patch_size
    data = (data + np.pi) / (2 * np.pi)
    if mask is not None:
        assert data.ndim == 4 and mask.ndim == 4, "number of input dims should be 4 got data: {} mask: {} instead".format(
        data.ndim, mask.ndim)
    reconstructed_intf = np.zeros((data.shape[0] * patch_H // stride + patch_H * (1 - 1 // stride),
                                   data.shape[1] * patch_W // stride + patch_W * (1 - 1 // stride)))

    reconstructed_pred = np.zeros((data.shape[0] * patch_H // stride + patch_H * (1 - 1 // stride),
                                   data.shape[1] * patch_W // stride + patch_W * (1 - 1 // stride)))

    if mask is not None:
        reconstructed_mask = np.zeros(
            (data.shape[0] * patch_H // stride + patch_H * (1 - 1 // stride),
             data.shape[1] * patch_W // stride + patch_W * (1 - 1 // stride)))


    if add_lidar_mask:
        lidar_mask_df = gpd.read_file('lidar_mask_polygs.shp')
        mask_polyg = lidar_mask_df[lidar_mask_df['source'] == intf_lidar_mask]
        height, width = reconstructed_intf.shape[0], reconstructed_intf.shape[1]  # Raster dimensions
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

        if plot:
            plt.imshow(rasterized_polygon)
            plt.show()
    for i in range(data.shape[0]):
        print(i)

        for j in range(data.shape[1]):
            reconstructed_intf[i * patch_H // stride: i * patch_H // stride + patch_H,
            j * patch_W // stride: j * patch_W // stride + patch_W] += data[
                                                                                               i, j] / stride ** 2
            if mask is not None:
                reconstructed_mask[i * patch_H // stride:i * patch_H // stride + patch_H,
                j * patch_W // stride: j * patch_W // stride + patch_W] += mask[
                                                                                               i, j] /stride ** 2

            # yr0 = y0 - dy * i * patch_H/args.data_stride
            # yrn = yr0 - dy * patch_H
            # xr0 = x4000 + dx *j * patch_W/args.data_stride
            # xrn = xr0 + dx * patch_W
            # rectangle = box(xr0, yrn, xrn, yr0)
            # rectangle_gdf = gpd.GeoDataFrame(geometry=[rectangle])
            #
            #
            # is_within_mask = True# mask_polyg.geometry.apply(lambda poly: rectangle.within(poly)).any()
            # ints_area = rectangle_gdf.intersection(mask_polyg).area
            # intersection_areas = []

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
            # if relative_intersection > 0.5:
            is_within_mask = True
            if add_lidar_mask:
                is_within_mask = np.all(rasterized_polygon[
                                        i * patch_H // stride:i * patch_H // stride + patch_H,
                                        j * patch_W // stride: j * patch_W // stride + patch_W])

            if is_within_mask:
                # if  np.any(data[i,j]>5):
                image = torch.tensor(data[i, j]).unsqueeze(0).unsqueeze(1).to(device=device,
                                                                              dtype=torch.float32,
                                                                              memory_format=torch.channels_last)
                pred = net(image)
                pred = (F.sigmoid(pred) > 0.5).float()
                pred = pred.squeeze(1).squeeze(0).cpu().detach().numpy()

                reconstructed_pred[i * patch_H // stride:i * patch_H // stride + patch_H,
                j * patch_W // stride: j * patch_W // stride + patch_W] += pred / stride ** 2

        # fig, (ax1,ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(data[i,j])
        # ax2.imshow(mask[i,j])
        # ax3.imshow(pred)
        # plt.show()
    reconstructed_pred = np.where(reconstructed_pred > rth, 1, 0).astype(np.float32)

    if mask is None:
        return reconstructed_intf,reconstructed_pred
    else:
        return reconstructed_intf,reconstructed_mask,reconstructed_pred

def get_pred_args():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')
    parser.add_argument('--full_intf_dir',  type=str, default='./', help='full interferogram path')
    parser.add_argument('--input_patch_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='patches inputs')
    parser.add_argument('--plot_data',  type=bool, default=False)
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--eleven_days_diff',  type=str, default='False', help='Flag to take only 11 days difference interferograms')

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

    parser.add_argument('--plot', type=str,default='False')
    parser.add_argument('--attn_unet',action='store_true')



    return parser.parse_args()



if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    logging.basicConfig(level=logging.INFO)  # Set the logging level (e.g., INFO)
    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)
    now = datetime.now().strftime("%m_%d_%Hh%M")

    args = get_pred_args()
    args.eleven_days_diff = str2bool(args.eleven_days_diff)
    args.plot = str2bool(args.plot)
    args.add_lidar_mask = str2bool(args.add_lidar_mask)
    if not is_running_locally:
        args.plot = False
    job_name = args.job_name +'_'+ now
    model_name = args.model.split('.')[0]
    output_path = 'pred_outputs2/' + model_name + '/' + job_name + '/'
    output_polyg_path = output_path + 'polygs/'
    try:
        os.makedirs(output_path)
        logging.info(f"Directory '{output_path}' created successfully")
    except FileExistsError:
        logging.info(f"Directory '{output_path}' already exists")
    try:
        os.makedirs(output_polyg_path)
        logging.info(f"Directory '{output_polyg_path}' created successfully")
    except FileExistsError:
        logging.info(f"Directory '{output_polyg_path}' already exists")

    log_file = output_path + args.job_name + '_' + now + '.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)

    logging.info('Running job {} with model {}.pth and args: {}'.format(job_name, model_name, args))

    if (args.intf_source == 'intf_list' and args.intf_list is None) or (args.intf_source == 'test_dataset' and args.test_dataset is None) or (args.intf_source == 'preset' and args.preset is None) :
        logging.info('you chose to take intfs from '+args.intf_source + ' but it is None. exiting.')
        sys.exit()

    if args.intf_source =='intf_list':
        intf_list = args.intf_list.split(',')

    elif args.intf_source == 'preset':
        with open(args.valset_from_partition, 'r') as file:
            loaded_data = json.load(file)
        intf_list = loaded_data['val']
        logging.info('taking test intfs from partition {}. intf list: {}'.format(args.valset_from_partition,intf_list))

    else:
        with open('./test_data/'+args.test_dataset, 'rb') as file:
            test_data = pickle.load(file)

        intf_list = test_data.ids
        logging.info('taking test intfs from test dataset {}. intf list: {}'.format(args.test_dataset, intf_list))

    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    if args.attn_unet:
        net = AttentionUNet(n_channels=1,n_classes=1, bilinear=False)
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
    data_dir = args.input_patch_dir + 'data_patches_H' + str(patch_H) + '_W' + str(patch_W)+'_strpp{}'.format(args.data_stride) + ('_11days' if args.eleven_days_diff else '_all')
    mask_dir = args.input_patch_dir + 'mask_patches_H' + str(patch_H) + '_W' + str(patch_W) + '_strpp{}'.format(args.data_stride) + ('_11days' if args.eleven_days_diff else '_all')

    for intf in intf_list:

        intfs_coords= get_intf_coords(intf)
        data_file_name = 'data_patches_' + intf + '_H' + str(patch_H) + '_W' + str(patch_W)+'_strpp{}'.format(args.data_stride) +'.npy'
        mask_file_name = 'mask_patches_' + intf + '_H' + str(patch_H) + '_W' + str(patch_W)+'_strpp{}'.format(args.data_stride) +'.npy'
        data_path = data_dir + '/' + data_file_name
        mask_path = mask_dir + '/' + mask_file_name
        data = np.load(data_path)
        mask = np.load(mask_path)
        reconstructed_intf,reconstructed_mask,reconstructed_pred = reconstruct_intf_prediction(data, intfs_coords, net,(patch_H,patch_W) ,args.data_stride, args.recon_th,mask)



        x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nz,bo = intfs_coords

        # Convert mask array to polygons with latlon coordinates
        polygons = plg_indx2longlat(mask_array_to_polygons(reconstructed_pred),intfs_coords)



        list_of_polygons = polygons.geometry.tolist()


        prefix = output_path  + intf
        out_polyg_path = output_polyg_path +   intf +'_predicted_polyogns.shp'
        polygons.to_file(out_polyg_path)


        np.save(prefix +'_pred', reconstructed_pred)
        np.save(prefix +'_image', reconstructed_intf)
        np.save(prefix +'_gt', reconstructed_mask)


        if args.plot:
            fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5),sharex=True,sharey=True)
            extent = [x4000, x4000+dx*reconstructed_intf.shape[1], y0 - dy * reconstructed_intf.shape[0], y0]


            ax1.imshow(reconstructed_intf,extent=extent)
            ax1.set_title('Reconstructed Image')
            for p in list_of_polygons:
                x,y = p.exterior.xy
                ax1.plot(x,y)

            ax2.imshow(reconstructed_mask,extent=extent)
            ax2.set_title('Reconstructed True mask')

            ax3.imshow(reconstructed_pred,extent=extent)
            ax3.set_title('pred mask')
            # def on_xlims_change(axes):
            #     for ax in (ax1, ax2, ax3):
            #         if ax != axes:
            #             ax.set_xlim(axes.get_xlim())
            #
            #
            # def on_ylims_change(axes):
            #     for ax in (ax1, ax2, ax3):
            #         if ax != axes:
            #             ax.set_ylim(axes.get_ylim())
            #
            #
            # # Connect the events
            # ax1.callbacks.connect('xlim_changed', on_xlims_change)
            # ax1.callbacks.connect('ylim_changed', on_ylims_change)

            # Show the plot###
            plt.show()





