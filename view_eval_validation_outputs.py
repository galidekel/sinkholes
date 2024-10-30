import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
import os
import argparse
import random


image = np.load('outputs/rand_by_intf/image_valid_test.npy', allow_pickle=True)
mask_pred = np.load('outputs/rand_by_intf/mask_pred_valid_epoch60.npy', allow_pickle=True)
mask_true = np.load('outputs/rand_by_intf/mask_true_valid.npy', allow_pickle=True)
numbers = list(range(1, image.shape[0]+1))
random.shuffle(numbers)
numbers = numbers[:30]

for i in numbers:
    show = random.randint(0,1)
    if show == 1:
        fig, [ax1,ax2,ax3] = plt.subplots(1,1)
        im = ax1.imshow(image[i])
        ax1.set_title('Input Patch')

        ax2.imshow(mask_true[i])
        ax2.set_title('GT')
        ax3.imshow(mask_pred[i])
        ax3.set_title('Pred')
        plt.show()
precision = precision1(mask_true,mask_pred)
recall = recall1(mask_true,mask_pred)
print('precision: {}'.format(precision),'\n recall: {}'.format(recall))
r, p = object_level_evaluate(mask_true,mask_pred)
print(' OL precision: {}'.format(p),'\n OL recall: {}'.format(r))
