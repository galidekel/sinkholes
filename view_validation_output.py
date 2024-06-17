import numpy as np
import matplotlib.pyplot as plt
from evaluate import *

#plt.rcParams['backend'] = 'Qt5Agg'

image = np.load('outputs/validation/spatial/11_day_diff/image_valid_test.npy', allow_pickle=True)
mask_pred = np.load('outputs/validation/spatial/11_day_diff/mask_pred_valid_epoch26.npy', allow_pickle=True)
mask_true = np.load('outputs/validation/spatial/11_day_diff/mask_true_valid.npy', allow_pickle=True)

precision,recall = calc_precision_recall(mask_pred,mask_true)
print(precision,recall)
precision = precision1(mask_true,mask_pred)
recall = recall1(mask_true,mask_pred)
print(precision,recall)
object_level_evaluate(mask_true,mask_pred)

for i in range(mask_pred.shape[0]):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(image[i])
    ax1.set_title('Image')
    ax2.imshow(mask_true[i])
    ax2.set_title('True mask')
    ax3.imshow(mask_pred[i])
    ax3.set_title('Predicted mask')
    plt.show()