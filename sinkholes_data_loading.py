import logging
import os
import sys

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
from torch.utils.data import Dataset , DataLoader
from tqdm import tqdm
import json
import math
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
    mask_file = list(mask_dir.glob('mask_patches_nonz_'+idx + mask_suffix))[0]
    mask = np.load(mask_file)
    if mask[0].ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class SubsiDataset(Dataset):
    def __init__(self,args, image_dir, mask_dir, intrfrgrm_list=None, scale: float = 1.0, mask_suffix: str = '',dset = 'train' ):
        super(SubsiDataset, self).__init__()

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        # if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
        #     logging.info('It seems that the data you are requesting does not exist, please check.')
        #     sys.exit(0)
        assert os.path.exists(self.image_dir) and os.path.exists(self.mask_dir), 'The data you are requesting does not exist'
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        if args.nonz_only and args.partition_mode != 'spatial':
          pref , mask_pref = 'data_patches_nonz_','mask_patches_nonz_'
        else:
          pref , mask_pref = 'data_patches_', 'mask_patches_'

        start_intf_name = len(pref)
        if args.partition_mode == 'spatial':
            with open(args.intf_dict_path, 'r') as json_file:
                coord_dict = json.load(json_file)

        if intrfrgrm_list is None:
            intrfrgrm_list = [file for file in listdir(image_dir) if ('nonz' in file and args.nonz_only and args.partition_mode!='spatial') or ('nonz' not in file and (not args.nonz_only or args.partition_mode == 'spatial'))]
        self.ids = [file.split('.')[0][start_intf_name:] for file in intrfrgrm_list]

        if not self.ids:
            raise RuntimeError(f'No input file found in {image_dir}, make sure you put your images there')
        self.image_data,self.mask_data,self.index_map = [],[],[]
        for i,id in enumerate(self.ids):
            if args.partition_mode != 'spatial':
                image_data = np.load(join(self.image_dir, pref + id + '.npy'))
                mask_data = np.load(join(self.mask_dir, mask_pref + id +'.npy'))
                if not args.nonz_only:
                    image_data = image_data.reshape(-1,image_data.shape[2],image_data.shape[3])
                    mask_data = mask_data.reshape(-1,mask_data.shape[2],mask_data.shape[3])
            else:

                intf_id = id[:17]
                stride = math.floor(args.patch_size[0]/2) *coord_dict[intf_id]["dY"]

                thresh_line = math.floor((coord_dict[intf_id]["north"] - args.thresh_lat) / stride)
                if thresh_line < 0:
                    thresh_line = 0
                if dset == 'train':
                    image_data = np.load(join(self.image_dir, pref + id + '.npy'))[:thresh_line]
                    mask_data = np.load(join(self.mask_dir, mask_pref + id + '.npy'))[:thresh_line]
                else:
                    image_data = np.load(join(self.image_dir, pref + id + '.npy'))[thresh_line:]
                    mask_data = np.load(join(self.mask_dir, mask_pref + id + '.npy'))[thresh_line:]
                image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
                mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])
                if args.nonz_only:
                    mask_nz, image_nz = [],[]
                    for p in range(image_data.shape[0]):
                        if np.any(mask_data[p]>0):
                            mask_nz.append(mask_data[p])
                            image_nz.append(image_data[p])
                    image_data = np.array(image_nz)
                    mask_data = np.array(mask_nz)
                    logging.info('Sptial partition: Created {} patches for {} set'.format(image_data.shape[0], dset))
            image_data = np.expand_dims(image_data,axis=0)
            mask_data = np.expand_dims(mask_data,axis=0)
            len_examples = image_data.shape[1]
            self.image_data.append(image_data)
            self.mask_data.append(mask_data)
            self.index_map.extend([[i,j] for j in range(len_examples)])

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix='.npy'), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
        logging.info(f'Created dataset: from {len(self.ids)} interferograms {len(self.index_map)} patches!!!')

    def __len__(self):
        return len(self.index_map)

    @staticmethod
    def preprocess(mask_values, img,  is_mask):


        if is_mask:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            # if img.ndim == 2:
            #     img = img[np.newaxis, ...]
            # else:
            #     img = img.transpose((2, 0, 1))
            img = (img +np.pi) / 2*np.pi


            return img

    def __getitem__(self, sample):
        intfgrm_num, patch_num= self.index_map[sample][0], self.index_map[sample][1]
        img = self.image_data[intfgrm_num][0][patch_num].astype(np.float32)
        msk = self.mask_data[intfgrm_num][0][patch_num].astype(np.float32)
        img = self.preprocess(self.mask_values,img,0)
        msk = self.preprocess(self.mask_values,msk,1)

        assert img.size == msk.size, \
            f'Image and mask should be the same size, but are {img.size} and {msk.size}'


        return {
            'image': torch.as_tensor(img.copy()).unsqueeze(0).float().contiguous(),
            'mask': torch.as_tensor(msk.copy()).long().contiguous()
        }
if __name__ == '__main__':
     intrfrgrm_list = None#['20230802T153919_20230813T153921']
     dataset = SubsiDataset('./data_patches', './mask_patches')
     train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True)
     print('train_loaderlength: ',len(train_loader))
     for batch, (input,targ) in enumerate(train_loader):
         print(batch)


