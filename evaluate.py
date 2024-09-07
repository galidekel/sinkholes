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

#plt.rcParams['backend'] = 'Qt5Agg'


def object_level_evaluate( gt, pred,epsilon = 1e-7):

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    intersect_recall, intersect_precision = [], []
    patch_gt_areas, patch_pred_areas = [], []
    for i in range(gt.shape[0]):
        transform = Affine.identity()  # Create an identity transform

        # Convert binary mask to Shapely polygons
        gt_polygons,pred_polygons = [], []


        for geom, val in rasterio.features.shapes(gt[i], transform=transform):
            if val > 0:  # Assuming binary mask has values 0 and 1
                gt_polygons.append(shape(geom))

        for geom, val in rasterio.features.shapes(pred[i], transform=transform):
            if val > 0:  # Assuming binary mask has values 0 and 1
                pred_polygons.append(shape(geom))

        th = 0.5
        buffer = 8
        if False:
            fig, (ax1, ax2,ax3) = plt.subplots(1, 3,sharex=True, sharey=True)


            ax1.imshow(gt[i], cmap='gray')
            # for item in gt_polygons:
            #     # x, y = item.buffer(buffer).exterior.xy
            #     ax1.plot(x, y)
            ax1.set_title('GT')

            ax2.imshow(pred[i], cmap='gray')

            ax2.set_title('PRED')

            ax3.imshow(pred[i], cmap='gray')
            for item in pred_polygons:
                x, y = item.exterior.xy
                ax3.plot(x, y)
            ax3.set_title('PRED')


            plt.show()

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
def evaluate(net, dataloader, device, amp ,is_local,out_path,epoch,mode = 'val'):
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
                mask_pred_np = mask_pred.squeeze(1).cpu().detach().numpy()
                mask_true_np = mask_true.squeeze(1).cpu().detach().numpy()
                image_np = image.squeeze(1).cpu().detach().numpy()
                epoch_suf = '_epoch' + str(epoch)
                if epoch % 2 == 0:
                    pred_batches.append(mask_pred_np)
                if epoch == 1:
                    image_batches.append(image_np)
                    true_mask_batches.append(mask_true_np)

                # plot for testing
                if is_local and mode == 'test':
                    import matplotlib.pyplot as plt
                    image_np = image.detach().numpy()
                    mask_pred_np = mask_pred.detach().numpy()
                    mask_true_np = mask_true.detach().numpy()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    sc1 = ax1.imshow(image[0,0,:,:])
                    ax1.set_title('Input Patch')
                    #cbar = fig.colorbar(sc1, ax=ax1,orientation='vertical',aspect = 20)

                    ax2.imshow(mask_true_np[0, 0, :, :])
                    ax2.set_yticks([])
                    ax2.set_title('True Mask')
                    ax3.imshow(mask_pred_np[0, 0, :, :])
                    ax3.set_yticks([])
                    ax3.set_title('Predicted Mask')

                    plt.show()
              #### plt for testing

                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                if mode == 'test':
                    batch_precision, batch_recall = calc_precision_recall(mask_pred_np, mask_true_np)
                    olr, olp, b_gt_a = object_level_evaluate(mask_true_np, mask_pred_np)
                    precision += batch_precision
                    recall += batch_recall
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


        if epoch == 1 and mode == 'val':
            image = np.concatenate(image_batches)
            mask_true = np.concatenate(true_mask_batches)
            np.save(out_path + '/image_valid_test', image)
            np.save(out_path + '/mask_true_valid', mask_true)

        if epoch % 2 == 0:
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
