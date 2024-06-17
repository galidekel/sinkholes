import glob

import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
import os
import argparse

def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg

parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')
parser.add_argument('--path',  type=str, default='pred_outputs/job_train_11days_preset_by_intf_20_05_13h45_2024-05-24_13:35checkpoint_epoch60/job_p200100_thp0_06_16_14h10/', help='reconstructed interferogram path')
parser.add_argument('--plot',  type=str, default='False')
args = parser.parse_args()
path=args.path
plot=str2bool(args.plot)
plt.rcParams['backend'] = 'Qt5Agg'
files_list = os.listdir(path)
intf_list = [file[:17] for file in files_list if file[-3:] !='log']
unique_intf_list = list(set(intf_list))

images,preds,gts, precisions,recalls = [],[],[],[],[]
for item in unique_intf_list:
    image = np.load( path + item +'_image.npy', allow_pickle=True)
    mask_pred = np.load(path + item+'_pred.npy', allow_pickle=True)
    mask_true = np.load(path + item + '_gt.npy', allow_pickle=True)

    if len(mask_pred.shape) <3:
        mask_pred = np.expand_dims(mask_pred, axis=0)
        mask_true = np.expand_dims(mask_true, axis=0)
        image = np.expand_dims(image, axis=0)
    images.append(image)
    preds.append(mask_pred)
    gts.append(mask_true)

    precision = precision1(mask_true,mask_pred)
    recall = recall1(mask_true,mask_pred)
    print(precision,recall)
    r, p = object_level_evaluate(mask_true,mask_pred)
    recalls.append(r)
    precisions.append(p)

    if plot:
        for i in range(mask_pred.shape[0]):
            fig, (ax1,ax2,ax3) = plt.subplots(1,3)
            ax1.imshow(image[i])
            ax1.set_title('Image')
            ax2.imshow(mask_true[i])
            ax2.set_title('True mask')
            ax3.imshow(mask_pred[i])
            ax3.set_title('Predicted mask')
            plt.show()
recall_np = np.array(recalls)
precision_np = np.array(precisions)

mean_Recall = np.mean(recall_np)
mean_Precision = np.mean(precision_np)
print('Mean Recall: ', mean_Recall)
print('Mean Precision: ', mean_Precision)