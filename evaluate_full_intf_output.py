import glob

import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
import os
import argparse
import geopandas as gpd
from get_intf_info import *
from view_reconstructed_pred import plg_longlat2indx
import json
from get_intf_info import *
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter


def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg

parser = argparse.ArgumentParser(description='evaluate full intf output')
parser.add_argument('--path',  type=str, default='pred_outputs/job_train_11days_preset_by_intf_20_05_13h45_2024-05-24_13h35checkpoint_epoch60/job_p200100_thp25_withlidarmask_06_26_18h00/', help='reconstructed interferogram path')
parser.add_argument('--path2',  type=str, default='pred_outputs/job_train_11days_preset_by_intf_20_05_13h45_2024-05-24_13h35checkpoint_epoch60/job_p200100_thp75_with_lidar_mask_07_04_16h29/', help='reconstructed interferogram path')
parser.add_argument('--path3',  type=str, default='pred_outputs/job_train_11days_preset_by_intf_20_05_13h45_2024-05-24_13h35checkpoint_epoch60/job_p200100_thp75_with_lidar_mask_07_04_16h29/', help='reconstructed interferogram path')

parser.add_argument('--plot',  type=str, default='False')
args = parser.parse_args()
path=args.path
path2=args.path2
path3=args.path3

plot=str2bool(args.plot)
#plt.rcParams['backend'] = 'Qt5Agg'
files_list = os.listdir(path)
intf_list = [file[:17] for file in files_list if (file[-3:] !='log' and '20210131' not in file and os.path.isfile(args.path + file))]
unique_intf_list = list(set(intf_list))
#recheck nonz num
with open('intf_coord.json', 'r') as json_file:
    coord_dict = json.load(json_file)

for intf in unique_intf_list:
    print (intf)
    print(coord_dict[intf]['north'])
    print(coord_dict[intf]['nonz_num'])
    print('---------------------')

images,preds,gts, precisions,recalls,precisions2,recalls2,precisions3,recalls3 = [],[],[],[],[],[],[],[],[]
for item in unique_intf_list:

        # if '0613' not in item and '1210' not in item:
        if '20220713' not in item:
             continue
        print(item)
        intf_info = get_intf_coords(item)
        x4000 = intf_info[6]
        dx = intf_info[2]
        y0 = intf_info[1]
        dy = intf_info[3]

        image = np.load( path + item +'_image.npy', allow_pickle=True)
        mask_pred = np.load(path + item+'_pred.npy', allow_pickle=True)
        mask_true = np.load(path + item + '_gt.npy', allow_pickle=True)
        mask_pred2 =  np.load(path2 + item+'_pred.npy', allow_pickle=True)
        mask_pred3 = np.load(path3 + item+'_pred.npy')
        # pred_polygs = gpd.read_file(path + 'polygs/' + item + '_predicted_polyogns.shp')
        # pred_polygs2 = gpd.read_file(path2 + 'polygs/' + item + '_predicted_polyogns.shp')
        if y0<31.5:
            half_size_y = image.shape[0]//2
            half_size_x = image.shape[1]//2
            image = image[100:half_size_y,:half_size_x]
            mask_pred = mask_pred[100:half_size_y,:half_size_x]
            mask_true = mask_true[100:half_size_y,:half_size_x]
            mask_pred2 = mask_pred2[100:half_size_y,:half_size_x]
            mask_pred3 = mask_pred3[100:half_size_y,:half_size_x]
        else:
            size_y = image.shape[0]
            fifth_size_y = image.shape[0]//5
            half_size_x = image.shape[1]//2
            y0 = y0-fifth_size_y*dy
            image = image[fifth_size_y:size_y-1000, :half_size_x]
            mask_pred = mask_pred[fifth_size_y:size_y-1000, :half_size_x]
            mask_true = mask_true[fifth_size_y:size_y-1000, :half_size_x]
            mask_pred2 = mask_pred2[fifth_size_y:size_y-1000, :half_size_x]
            mask_pred3 = mask_pred3[fifth_size_y:size_y-1000, :half_size_x]


        if len(mask_pred.shape) <3:
            mask_pred = np.expand_dims(mask_pred, axis=0)
            mask_pred2 = np.expand_dims(mask_pred2, axis=0)
            mask_pred3 = np.expand_dims(mask_pred3,axis=0)
            mask_true = np.expand_dims(mask_true, axis=0)
            image = np.expand_dims(image, axis=0)
        images.append(image)
        preds.append(mask_pred)
        gts.append(mask_true)
        if False:
            precision = precision1(mask_true,mask_pred)
            recall = recall1(mask_true,mask_pred)
            print('intf ' + item + ':')
            #print(precision,recall)
            r, p,n = object_level_evaluate(mask_true,mask_pred,th=0.5,buffer=10)
            r2,p2,n = object_level_evaluate(mask_true,mask_pred2,th =0.5,buffer = 10)
            r3,p3,n = object_level_evaluate(mask_true,mask_pred3,th=0.5,buffer = 10)
            recalls.append(r)
            precisions.append(p)
            recalls2.append(r2)
            precisions2.append(p2)
            recalls3.append(r3)
            precisions3.append(p3)

        plot = True
        if plot:

            for i in range(mask_pred.shape[0]):
                extent = [x4000, x4000 + dx * image[i].shape[1], y0 - dy * image[i].shape[0], y0]

                fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,sharex=True,sharey=True,figsize=(9,7))
                #fig.suptitle(item)
                ax1.imshow(image[i],extent=extent,cmap='jet')
                ax1.set_title('Original \n Interferogram', fontsize=10)
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #ax1.set_xticks(np.linspace(extent[0], extent[1], 2))  # 5 evenly spaced ticks on x-axis

                ax1.tick_params(axis='x', labelsize=9)
                ax1.tick_params(axis='y', labelsize=9)


                #cbar = fig.colorbar(im, ax=ax1)
                # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
                # cbar.ax.tick_params(labelsize=5)  # Set the label size to 8

                #pred_polygs.plot(ax=ax1, edgecolor='b', facecolor='none')
                cmap = colors.ListedColormap(['purple', 'yellow'])
                cmap = colors.ListedColormap(['white', 'black'])


                ax2.imshow(mask_true[i]*255,extent=extent)
                ax2.set_title('True Mask \n', fontsize=10)
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                ax2.tick_params(axis='x', labelsize=9)

                ax3.imshow(mask_pred[i],extent=extent)
                ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                ax3.tick_params(axis='x', labelsize=9)

                ax3.set_title('Predicted Mask \n RTh 0.0', fontsize=10)
                ax4.imshow(mask_pred2[i],extent=extent)
                ax4.set_title('Predicted Mask \n RTh 0.25',fontsize=10)
                ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                ax4.tick_params(axis='x', labelsize=9)

                ax5.imshow(mask_pred3[i], extent=extent)
                ax5.set_title('Predicted Mask \n RTh 0.5', fontsize=10)
                ax5.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                ax5.tick_params(axis='x', labelsize=9 )


                # pred_polygs.plot(ax=ax3, edgecolor='b', facecolor='none')

                fig.savefig('/Users/galidek/Desktop/paper_figs/full_intfs/'+item, dpi=1000)
                plt.show()
recall_np = np.array(recalls)
precision_np = np.array(precisions)
recall2_np = np.array(recalls2)
precision2_np = np.array(precisions2)
recall3_np = np.array(recalls3)
precision3_np = np.array(precisions3)

mean_Recall = np.mean(recall_np)
mean_Precision = np.mean(precision_np)
mean_Recall2 = np.mean(recall2_np)
mean_Precision2 = np.mean(precision2_np)
mean_Recall3 = np.mean(recall3_np)
mean_Precision3 = np.mean(precision3_np)
print('Mean Recall: ', mean_Recall)
print('Mean Precision: ', mean_Precision)
print('Mean Recall2: ', mean_Recall2)
print('Mean Precision2: ', mean_Precision2)
print('Mean Recall3: ', mean_Recall3)
print('Mean Precision3: ', mean_Precision3)