import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_score import multiclass_dice_coeff, dice_coeff

import numpy as np
from affine import Affine
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import shapes

import rasterio.features
from shapely.geometry import shape

import geopandas as gpd
from unet import *
import logging

plt.rcParams['backend'] = 'Qt5Agg'


def object_level_evaluate( gt, pred,epsilon = 1e-7,th=0.7,buffer=5):

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    intersect_recall, intersect_precision = [], []
    patch_gt_areas, patch_pred_areas = [], []
    for i in range(gt.shape[0]):
        transform = Affine.identity()  # Create an identity transform

        # Convert binary mask to Shapely polygons
        gt_polygons,pred_polygons = [], []

        ###create polygons
        for geom, val in rasterio.features.shapes(gt[i], transform=transform):
            if val > 0:  # Assuming binary mask has values 0 and 1
                gt_polygons.append(shape(geom))

        for geom, val in rasterio.features.shapes(pred[i], transform=transform):
            if val > 0:  # Assuming binary mask has values 0 and 1
                pred_polygons.append(shape(geom))

        ###start calculation for patch
        intersection_area= 0
        total_pred_area = sum(pp.area for pp in pred_polygons)
        total_gt_area = sum(pgt.area for pgt in gt_polygons)

        patch_gt_areas.append(total_gt_area)
        patch_pred_areas.append(total_pred_area)
        for pgt in gt_polygons:
            intersection = 0
            for pp in pred_polygons:
                buffered_pp = pp.buffer(buffer)
                intersection += pgt.intersection(buffered_pp).area/pgt.area
                if False:
                    x, y = pp.exterior.xy
                    x1, y1 = buffered_pp.exterior.xy
                    x2, y2 = pgt.exterior.xy
                    ints = pgt.intersection(buffered_pp)

                    if not ints.is_empty:
                        x_int, y_int = ints.exterior.xy

                        fig, [ax1,ax2] = plt.subplots(1, 2, sharex=True, sharey=True)

                        ax1.imshow(gt[i], cmap='gray')
                        ax1.set_title('Ground Truth Mask')
                        ax2.imshow(pred[i], cmap='gray')
                        ax2.set_title('Predicted Mask')
                        ax2.plot(x, y,color='red',linewidth=2)
                        ax2.plot(x1, y1, color='blue',linewidth=2)
                        ax2.plot(x2, y2,color='green',linewidth=2)
                        ax2.fill(x_int, y_int, alpha=0.5, fc='yellow', label='Intersection')


                        plt.show()
            if intersection > th:
                intersection_area += pgt.area
        ol_intersection_recall = round(intersection_area / (total_gt_area + epsilon),2)
        ol_intersection_precision = round(intersection_area / (total_pred_area + epsilon),2)
        if ol_intersection_precision  > 1:
            ol_intersection_precision =1.0

        intersect_recall.append(ol_intersection_recall)
        intersect_precision.append(ol_intersection_precision)

        # print('{}: object level precision score:'.format(i), ol_intersection_precision)
        # print('object level recall score:', ol_intersection_recall)

    batch_gt_area = np.sum(np.array(patch_gt_areas))
    ol_batch_recall = round(np.sum(np.array(intersect_recall)*np.array(patch_gt_areas))/batch_gt_area,2)
    ol_batch_precision = round(np.sum(np.array(intersect_precision)*np.array(patch_gt_areas))/batch_gt_area,2)
    print('ol batch recall:',ol_batch_recall)
    print('ol batch precision:',ol_batch_precision)

    return ol_batch_recall, ol_batch_precision,batch_gt_area

def precision1(gt, pred, epsilon=1e-7):
    # Convert arrays to boolean arrays
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # Calculate true positives (TP), false positives (FP) for the entire batch
    TP = np.logical_and(gt, pred).sum().astype(float)
    FP = np.logical_and(~gt, pred).sum().astype(float)

    # Calculate precision for the entire batch
    precision = TP / (TP + FP + epsilon)

    return precision
def recall1(gt, pred, epsilon=1e-7):
    # Convert arrays to boolean arrays
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # Calculate true positives (TP), false negatives (FN) for the entire batch
    TP = np.logical_and(gt, pred).sum().astype(float)
    FN = np.logical_and(gt, ~pred).sum().astype(float)

    # Calculate recall for the entire batch
    recall = TP / (TP + FN + epsilon)

    return recall
def calc_precision_recall (y_pred, y_true):
    precision = np.sum(y_pred * y_true)/np.sum(y_pred) #tp/tp+tn
    recall = np.sum(y_pred * y_true)/np.sum(y_true) #tp/tp+fn
    return precision, recall

@torch.inference_mode()
def evaluate(net, dataloader, device, amp ,is_local,out_path,epoch,mode = 'val',save_val = False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    precision = 0
    recall = 0
    ol_precision = 0
    ol_recall = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        image_batches,true_mask_batches, pred_batches, b_gts = [],[],[],[]
        for bn, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_true = mask_true.unsqueeze(1)
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                mask_pred_np = mask_pred.squeeze(1).cpu().detach().numpy() #squeeze to (b,H,W) when saving npy
                mask_true_np = mask_true.squeeze(1).cpu().detach().numpy()
                image_np = image.squeeze(1).cpu().detach().numpy()
                epoch_suf = '_epoch' + str(epoch)
                if epoch % 2 == 0:
                    pred_batches.append(mask_pred_np)
                if epoch == 1:
                    image_batches.append(image_np)
                    true_mask_batches.append(mask_true_np)

                # plot for testing
                if is_local and False: #and mode == 'test':
                    import matplotlib.pyplot as plt

                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
                    sc1 = ax1.imshow(image_np[0,:,:], cmap='jet')
                    ax1.set_title('Input Patch')
                    ax1.set_yticks([0,100,200])
                    ax1.set_xticks([0,50,100])


                    # cbar = fig.colorbar(sc1, ax=ax1,orientation='vertical',aspect = 20, ticks=[0, 0.25, 0.5, 0.75, 1.0])
                    #
                    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax2.imshow(mask_true_np[0, :, :])
                    ax2.set_yticks([0,100,200])
                    ax2.set_xticks([0,50,100])

                    ax2.set_title('True Mask')
                    ax3.imshow(mask_pred_np[0, :, :])
                    ax3.set_yticks([0, 100, 200])
                    ax3.set_xticks([0,50,100])

                    ax3.set_title('Predicted Mask')


                    fig_name = 'byspatial_ccc'+str(bn)
                    plt.savefig(f'/Users/galidek/Desktop/paper_figs/nonz_patches/{fig_name}')
                    plt.show()
              #### plt for testing

                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                if mode == 'test':
                    batch_precision, batch_recall = calc_precision_recall(mask_pred_np, mask_true_np)
                    olr, olp, b_gt_a = object_level_evaluate(mask_true_np, mask_pred_np,th=0.7,buffer=5)
                    precision += batch_precision
                    recall += batch_recall
                    if not np.isnan(olp) and not np.isnan(olr):
                        ol_precision += olp * b_gt_a
                        ol_recall +=olr * b_gt_a
                        b_gts.append(b_gt_a)

            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)


        if epoch == 1 and mode == 'val' and save_val:
            image = np.concatenate(image_batches)
            mask_true = np.concatenate(true_mask_batches)
            np.save(out_path + '/image_valid_test', image)
            np.save(out_path + '/mask_true_valid', mask_true)

        if epoch % 2 == 0  and save_val:
            mask_pred = np.concatenate(pred_batches)
            np.save(out_path + '/mask_pred_valid'+epoch_suf,mask_pred)
    net.train()
    mean_dice_score = dice_score / max(num_val_batches, 1)
    if mode == 'test':
        mean_p = round(precision / num_val_batches,2)
        mean_r = round(recall / num_val_batches,2)
        mean_ol_p = round(ol_precision/sum(b_gts),2)
        mean_ol_r = round(ol_recall / sum(b_gts),2)

        print('mean dice score: ', mean_dice_score)
        print('Mean pixel level Precision: {}'.format(mean_p))
        print('Mean pixel level Recall: {}'.format(mean_r))

        print('Mean OL Precision: {}'.format(mean_ol_p))
        print('Mean OL Recall: {}'.format(mean_ol_r))
    return mean_dice_score
