import logging
import os
import sys
import random
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import math
import albumentations as A
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob('mask_patches_nonz_' + idx + mask_suffix))[0]
    mask = np.load(mask_file)
    if mask[0].ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def has_consecutive_zeros(arr, min_consecutive=10, max_consecutive=1000):
    # Check rows
    row_check = np.apply_along_axis(
        lambda x: np.any(np.convolve(x == 0, np.ones(min_consecutive, dtype=int), 'valid') == min_consecutive),
        axis=1,
        arr=arr
    )

    # Check columns
    col_check = np.apply_along_axis(
        lambda x: np.any(np.convolve(x == 0, np.ones(min_consecutive, dtype=int), 'valid') == min_consecutive),
        axis=0,
        arr=arr
    )

    return np.any(row_check) or np.any(col_check)


class SubsiDataset(Dataset):
    test_dataset_for_nonoverlap_split = []
    test_mask_for_nonoverlap_split = []

    def __init__(self, args, image_dir, mask_dir, intrfrgrm_list=None, scale: float = 1.0, mask_suffix: str = '',
                 dset='train', augment=False, seq_dict=None):
        super(SubsiDataset, self).__init__()

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        assert os.path.exists(self.image_dir) and os.path.exists(
            self.mask_dir), 'The data you are requesting does not exist'
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        if args.partition_mode == 'preset_by_intf':
            assert intrfrgrm_list is not None, "Partition mode 'preset by intf' requires an input intrfrgrm list"
        self.scale = scale
        self.mask_suffix = mask_suffix
        if args.add_temporal:
            self.temporal = True
            with open(image_dir + '/' + 'nonz_indices.json', "r") as f:
                nonz_pathces_dict = json.load(f)
                self.seq_dict = seq_dict
        else:
            self.temporal = False
        if args.nonz_only and args.partition_mode != 'spatial' and not args.add_nulls_to_train and not args.nonoverlap_tr_tst and not self.temporal:
            pref, mask_pref = 'data_patches_nonz_', 'mask_patches_nonz_'
        else:
            pref, mask_pref = 'data_patches_', 'mask_patches_'
        if args.use_cleaned_patches:
            suff, mask_suff = '_cleaned', '_cleaned'
        else:
            suff, mask_suff = '', ''
        start_intf_name = len(pref)
        if args.partition_mode == 'spatial':
            with open(args.intf_dict_path, 'r') as json_file:
                coord_dict = json.load(json_file)

        if intrfrgrm_list is None:
            self.ids = [file.split('.')[0][start_intf_name:start_intf_name + 17] for file in listdir(image_dir) if
                        ('nonz' in file and args.nonz_only and args.partition_mode != 'spatial') or (
                                    'nonz' not in file and (not args.nonz_only or args.partition_mode == 'spatial'))]
        else:
            self.ids = intrfrgrm_list
        if (args.test == 0 and dset == 'test'):
            self.index_map = []
            return

        if not self.ids:
            raise RuntimeError(f'empty set when trying to takr data from {image_dir}, recheck!!')

        self.image_data, self.mask_data, self.index_map = [], [], []
        test_data, test_mask_data = [], []
        for i, id in enumerate(self.ids):

            if args.partition_mode != 'spatial':

                image_data = np.load(join(self.image_dir, pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                    args.patch_size[1]) + '_strpp{}'.format(args.stride) + suff + '.npy'))
                mask_data = np.load(join(self.mask_dir,
                                         mask_pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                             args.patch_size[1]) + '_strpp{}'.format(args.stride) + mask_suff + '.npy'))
                if self.temporal:
                    rc = nonz_pathces_dict[id]  # [(x,y), ...] from current
                    tids = list(self.seq_dict[id]["prevs"]) + [id]  # prevs…present (T = k_prev+1)

                    def img_path(tid):
                        return join(self.image_dir,
                                    pref + tid + f"_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}{suff}.npy")

                    def msk_path(tid):
                        return join(self.mask_dir, pref.replace("data_",
                                                                "mask_") + tid + f"_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}{suff}.npy")

                    # load per-time images (current already in memory as image_data; same for mask_data if you have it)
                    img_pa = [image_data if tid == id else np.load(img_path(tid)).astype(np.float32) for tid in
                              tids]  # each (X,Y,H,W)
                    msk_pa = [mask_data if tid == id else np.load(msk_path(tid)).astype(np.float32) for tid in
                              tids]  # each (X,Y,H,W)

                    # keep only patch coords valid for **all** times
                    rc_valid = [(x, y) for (x, y) in rc if
                                all(0 <= x < p.shape[0] and 0 <= y < p.shape[1] for p in img_pa)]

                    # stack images per time: list[(N,H,W)] → (T,N,H,W)
                    patches_per_t = [np.stack([p[x, y] for (x, y) in rc_valid], axis=0) for p in
                                     img_pa]  # list of (N,H,W)
                    image_data = np.stack(patches_per_t, axis=0).astype(np.float32)  # (T,N,H,W)

                    # stack masks per time, then union over time: (T,N,H,W) → (N,H,W)
                    masks_per_t = [np.stack([p[x, y] for (x, y) in rc_valid], axis=0) for p in
                                   msk_pa]  # list of (N,H,W)
                    masks_TNHW = np.stack(masks_per_t, axis=0).astype(np.float32)  # (T,N,H,W)
                    mask_data = (masks_TNHW > 0).any(axis=0).astype(np.float32)

                elif args.retrain_with_fpz:
                    fpz_path = join(self.image_dir, 'additional_fp_patches/' + 'data_patches_fp_' + id + '.npy')
                    if os.path.isfile(fpz_path):

                        fpz_data = np.load(
                            join(self.image_dir, 'additional_fp_patches/' + 'data_patches_fp_' + id + '.npy'))
                        z_mask_data = np.zeros(fpz_data.shape)
                        image_data = np.concatenate((image_data, fpz_data), axis=0)
                        mask_data = np.concatenate((mask_data, z_mask_data), axis=0)
                    else:
                        logging.info('Note! No fpz patches for intf {}'.format(id))
                elif args.nonoverlap_tr_tst:
                    if dset == 'train':
                        nz_patches, nz_masks, nz_indices = [], [], []
                        for ii in range(image_data.shape[0]):
                            for jj in range(image_data.shape[1]):
                                if np.any(mask_data[ii, jj] > 0):
                                    nz_patches.append(image_data[ii, jj])
                                    nz_masks.append(mask_data[ii, jj])
                                    nz_indices.append([ii, jj])

                        test_num = int(args.test / 100 * len(nz_indices))
                        test_patch_nums = random.sample(range(len(nz_indices)), test_num)
                        test_patch_nums = sorted(test_patch_nums)
                        test_patches = [nz_patches[p] for p in test_patch_nums]
                        test_mask_patches = [nz_masks[p] for p in test_patch_nums]
                        test_indices = [nz_indices[p] for p in test_patch_nums]

                        train_patches, tr_mask_patches = [], []
                        for n, ind in enumerate(nz_indices):

                            if not (any((abs(item[0] - ind[0]) < 2 and abs(item[1] - ind[1]) < 2) for item in
                                        test_indices)):  ##only for stride 2!!
                                train_patches.append(nz_patches[n])
                                tr_mask_patches.append(nz_masks[n])

                        image_data, mask_data = np.array(train_patches), np.array(tr_mask_patches)
                        SubsiDataset.test_dataset_for_nonoverlap_split.append(np.array(test_patches))
                        SubsiDataset.test_mask_for_nonoverlap_split.append(np.array(test_mask_patches))
                    elif dset == 'test':
                        image_data, mask_data = SubsiDataset.test_dataset_for_nonoverlap_split[i], \
                        SubsiDataset.test_mask_for_nonoverlap_split[i]

                elif not args.nonz_only or args.add_nulls_to_train:
                    image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
                    mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])

                    if args.nonz_only:
                        image_patches = []
                        mask_patches = []

                        for j in range(image_data.shape[0]):
                            image_patch = image_data[j]
                            mask_patch = mask_data[j]
                            add_null_patch = False
                            if random.choice([True, False]) and not np.any(mask_patch > 0):
                                count = np.sum(image_patch == 0) < 1000
                                cons_zeros = has_consecutive_zeros(image_patch)
                                if count and cons_zeros:
                                    add_null_patch = True

                            if np.any(mask_patch > 0) or (add_null_patch):
                                image_patches.append(image_patch)
                                mask_patches.append(mask_patch)

                        print('number of patches with nulls: {}'.format(len(image_patches)))
                        print('from {} patches'.format(image_data.shape[0]))
                        image_data = np.array(image_patches)
                        mask_data = np.array(mask_patches)


            else:

                stride = math.floor(args.patch_size[0] / 2) * coord_dict[id]["dy"]

                thresh_line = math.floor((coord_dict[id]["north"] - args.thresh_lat) / stride)
                if thresh_line < 0:
                    thresh_line = 0
                if dset == 'train':
                    image_data = np.load(join(self.image_dir,
                                              pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                                  args.patch_size[1]) + '_strpp{}'.format(args.stride) + '.npy'))[
                                 :thresh_line]
                    mask_data = np.load(join(self.mask_dir,
                                             mask_pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                                 args.patch_size[1]) + '_strpp{}'.format(args.stride) + '.npy'))[
                                :thresh_line]
                else:
                    image_data = np.load(join(self.image_dir,
                                              pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                                  args.patch_size[1]) + '_strpp{}'.format(args.stride) + '.npy'))[
                                 thresh_line:]
                    mask_data = np.load(join(self.mask_dir,
                                             mask_pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                                 args.patch_size[1]) + '_strpp{}'.format(args.stride) + '.npy'))[
                                thresh_line:]
                image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
                mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])
                if args.nonz_only:
                    mask_nz, image_nz = [], []
                    for p in range(image_data.shape[0]):
                        if np.any(mask_data[p] > 0):
                            mask_nz.append(mask_data[p])
                            image_nz.append(image_data[p])
                    image_data = np.array(image_nz)
                    mask_data = np.array(mask_nz)
                    logging.info('Sptial partition: Created {} patches for {} set'.format(image_data.shape[0], dset))

            if not self.temporal:
                image_data = np.expand_dims(image_data, axis=0)
            mask_data = np.expand_dims(mask_data, axis=0)
            len_examples = image_data.shape[1]

            self.image_data.append(image_data)
            self.mask_data.append(mask_data)
            self.index_map.extend([[i, j] for j in range(len_examples)])
            logging.info('loaded patch for intf {}'.format(id))

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir,
                               mask_suffix='_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                   args.patch_size[1]) + '_strpp{}'.format(args.stride) + mask_suff + '.npy'),
                       self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

        self.do_augmentations = (args.augment and dset == 'train')
        if self.do_augmentations:
            self.augment = A.Compose([
                A.Rotate(limit=30, p=0.4),
                A.ElasticTransform(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.HorizontalFlip(p=0.3),
                A.GridDistortion(p=0.2),
            ], additional_targets={'mask': 'mask'})

        logging.info(f'Created dataset: from {len(self.ids)} interferograms {len(self.index_map)} patches!!!')

    def __len__(self):
        return len(self.index_map)

    @staticmethod
    def preprocess(mask_values, img, is_mask):

        if is_mask:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        out = img.copy()
        tol = 1e-1
        for c in range(out.shape[0]):
            mn = float(np.nanmin(out[c]));
            mx = float(np.nanmax(out[c]))
            if (mn < -tol) or (mx > 1.0 + tol):
                out[c] = (out[c] + np.pi) / (2 * np.pi)
            else:
                zmask = (out[c] == 0.0)  # only exact zeros
                if zmask.any():
                    out[c][zmask] = 0.5
        return out

        return img

    def __getitem__(self, sample):
        intf_idx, patch_idx = self.index_map[sample]

        if getattr(self, "temporal", False):
            # image: [T, N, H, W] -> pick patch -> [T, H, W]
            img = self.image_data[intf_idx][:, patch_idx].astype(np.float32)
            # mask:  [1, N, H, W] -> pick patch & drop singleton -> [H, W]
            msk = self.mask_data[intf_idx][0, patch_idx].astype(np.float32)
        else:
            # image/mask stored as [1, N, H, W] -> pick channel 0 & patch -> [H, W]
            img = self.image_data[intf_idx][0, patch_idx].astype(np.float32)
            msk = self.mask_data[intf_idx][0, patch_idx].astype(np.float32)

        # Preprocess
        img = self.preprocess(self.mask_values, img, 0)  # keeps shape; (T,H,W) or (H,W)
        msk = self.preprocess(self.mask_values, msk, 1)  # -> (H,W) labels
        T = img.shape[0];
        # fig, ax = plt.subplots(1, T + 1, figsize=(2 * (T + 1), 2))
        # for i in range(T): ax[i].imshow(img[i], cmap='gray'); ax[i].axis('off')
        # ax[-1].imshow(msk, cmap='Reds');
        # ax[-1].axis('off');
        # plt.tight_layout();
        # plt.show()

        # Augment (Albumentations expects channels-last)
        if self.do_augmentations:
            if getattr(self, "temporal", False):
                aug = self.augment(image=img.transpose(1, 2, 0), mask=msk)  # (H,W,T)
                img = aug["image"].transpose(2, 0, 1)  # back to (T,H,W)
                msk = aug["mask"]
            else:
                aug = self.augment(image=img, mask=msk)  # both (H,W)
                img = aug["image"]
                msk = aug["mask"]

        # Sanity: spatial sizes must match
        assert img.shape[-2:] == msk.shape[-2:], \
            f"Spatial size mismatch: image {img.shape} vs mask {msk.shape}"

        # To tensors
        if getattr(self, "temporal", False):
            # (T,H,W)
            img_t = torch.as_tensor(img.copy()).float().contiguous()
        else:
            # (1,H,W)
            img_t = torch.as_tensor(img.copy()).unsqueeze(0).float().contiguous()

        msk_t = torch.as_tensor(msk.copy()).long().contiguous()

        return {"image": img_t, "mask": msk_t}


if __name__ == '__main__':
    intrfrgrm_list = None  # ['20230802T153919_20230813T153921']
    dataset = SubsiDataset('./data_patches', './mask_patches')
    train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True)
    print('train_loaderlength: ', len(train_loader))
    for batch, (input, targ) in enumerate(train_loader):
        print(batch)

