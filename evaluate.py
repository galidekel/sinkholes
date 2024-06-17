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

def object_level_evaluate(gt, pred,epsilon = 1e-7):

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

        th = 0.7
        buffer = 5
        if i == 70:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(gt[i], cmap='gray')
            for item in gt_polygons:
                x, y = item.exterior.xy
                ax1.plot(x, y)


            ax2.imshow(pred[i], cmap='gray')
            for item in pred_polygons:
                x, y = item.exterior.xy
                ax2.plot(x, y)
                x, y = item.buffer(buffer).exterior.xy
                ax2.plot(x, y)

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


    batch_intersect_recall = round(np.sum(np.array(intersect_recall)*np.array(patch_gt_areas))/np.sum(np.array(patch_gt_areas)),2)
    batch_intersect_precision = round(np.sum(np.array(intersect_precision)*np.array(patch_gt_areas))/np.sum(np.array(patch_gt_areas)),2)
    print('batch intersect recall:',batch_intersect_recall)
    print('batch intersect precision:',batch_intersect_precision)

    return batch_intersect_recall, batch_intersect_precision

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
def evaluate(net, dataloader, device, amp ,is_local,out_path,epoch):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    precision = 0
    recall = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
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
                    np.save(out_path+'/mask_pred_valid'+epoch_suf,mask_pred_np)
                if epoch == 1:
                    np.save(out_path+'/image_valid_test',image_np)
                    np.save(out_path+'/mask_true_valid',mask_true_np)

                # plot for testing
                if is_local and False:
                    import matplotlib.pyplot as plt
                    image_np = image.detach().numpy()
                    mask_pred_np = mask_pred.detach().numpy()
                    mask_true_np = mask_true.detach().numpy()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.imshow(image[0,0,:,:])
                    ax2.imshow(mask_pred_np[0, 0, :, :])
                    ax2.set_title('predicted mask')
                    ax3.imshow(mask_true_np[0, 0, :, :])
                    ax3.set_title('true mask')
                    plt.show()
              #### plt for testing

                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                batch_precision, batch_recall = calc_precision_recall(mask_pred_np, mask_true_np)
                precision += batch_precision
                recall += batch_recall




            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
