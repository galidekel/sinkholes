import argparse
import logging
import os
import random
import sys

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
#from data_loading import *
#from mac_gpu import *
import io
from get_intf_info import *
import os
from torch.utils.data import Subset


# import wandb
from evaluate import evaluate
from unet import UNet
from sinkholes_data_loading import *
import logging
from dice_score import dice_loss

from datetime import datetime
import pickle
from attn_unet import *
import torch.nn.functional as F

EPS = 1e-6

def masked_bce_with_logits(logits, y_float, V, pos_w: float = 1.0):
    """
    logits:  (B,1,H,W)
    y_float: (B,1,H,W) in {0,1}
    V:       (B,1,H,W) validity mask in {0,1}
    pos_w:   >1.0 increases loss on positive pixels
    """
    pw = torch.as_tensor([pos_w], device=logits.device, dtype=logits.dtype)  # per-class (C=1)
    per_pix = F.binary_cross_entropy_with_logits(
        logits, y_float, reduction='none', pos_weight=pw
    )
    w = V.detach()
    return (per_pix * w).sum() / w.sum().clamp(min=1.0)

def masked_dice_loss_binary(logits, y_float, V):
    """
    Soft Dice on probs with validity mask.
    """
    p = torch.sigmoid(logits)                      # (B,1,H,W)
    num = (2.0 * (p * y_float * V)).sum()
    den = ((p * V).sum() + (y_float * V).sum() + EPS)
    return 1.0 - (num / den)

def masked_ce_plus_softdice_multiclass(logits, y_long, V):
    """
    CrossEntropy (masked) + mean soft Dice over classes (masked).
    logits: (B,C,H,W), y_long: (B,H,W), V: (B,1,H,W)
    """
    # CE (per-pixel) then mask
    ce_per_pix = F.cross_entropy(logits, y_long, reduction='none')   # (B,H,W)
    ce = (ce_per_pix * V.squeeze(1)).sum() / (V.sum().clamp(min=1.0))

    # Soft Dice over classes
    probs = F.softmax(logits, dim=1)                      # (B,C,H,W)
    y1 = F.one_hot(y_long, num_classes=logits.shape[1]).permute(0,3,1,2).float()  # (B,C,H,W)

    probs = probs * V                                     # mask both
    y1    = y1 * V

    num = (2.0 * (probs * y1)).sum(dim=(0,2,3))           # per-class
    den = (probs.sum(dim=(0,2,3)) + y1.sum(dim=(0,2,3)) + EPS)
    dice = 1.0 - (num / den).mean()                       # mean over classes

    return ce + dice

def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg
def train_model(
        args,
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        test_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0

):
    # 1. Create dataset
    patch_size = tuple(args.patch_size)
    H, W = patch_size

    image_dir = args.patches_dir + 'data_patches_H' + str(H) + '_W' + str(W) +'_strpp'+str(args.stride) + ('_11days' if args.train_on_11d_diff else '_all')
    mask_dir = args.patches_dir + 'mask_patches_H' + str(H) + '_W' + str(W)  +'_strpp'+str(args.stride) + ('_11days' if args.train_on_11d_diff else '_all')
    #
    if args.add_temporal or True:
        image_dir += '_Aligned'
        mask_dir += '_Aligned'

    if args.use_cleaned_patches:
        image_dir += '/cleaned'
        mask_dir += '/cleaned'

        #sys.exit(0)
    logging.info('input patch directories: {} and {}'.format(image_dir, mask_dir))
    assert os.path.exists(image_dir) and os.path.exists(image_dir), 'The data you are requesting does not exist, please check if you prepared it at the preparation stage'
    if args.nonz_only and args.partition_mode != 'spatial' and not args.add_temporal:
        pref, mask_pref = 'data_patches_nonz_', 'mask_patches_nonz_'
    else:
        pref, mask_pref = 'data_patches_', 'mask_patches_'
    start_intf_name = len(pref)
    intf_list = [
        file.split('.')[0][start_intf_name:start_intf_name + 17]
        for file in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, file)) and '.json' not in file and (
                ('nonz' in file and args.nonz_only and args.partition_mode != 'spatial'and not args.add_temporal) or
                ('nonz' not in file and (not args.nonz_only or args.partition_mode == 'spatial'or args.add_temporal))
        )
    ]
    with open(args.intf_dict_path, 'r') as json_file:
        coord_dict = json.load(json_file)
    tmp_list = []
    for intf in intf_list:
        if coord_dict[intf]['nonz_num'] != 'none':
            tmp_list.append(intf)
    intf_list = tmp_list

    n1 = len(intf_list)
    logging.info('Original list has {} nonz'.format(n1))
    if args.train_with_nonz_th:
        nonz_th_north, nonz_th_south = tuple(args.nonz_th)[0], tuple(args.nonz_th)[1]
        filtered_list = []
        for intf in intf_list:
            if (coord_dict[intf]['north'] > 31.5 and coord_dict[intf]['nonz_num'] > nonz_th_north) or (
                    coord_dict[intf]['north'] < 31.5 and coord_dict[intf]['nonz_num'] > nonz_th_south):
                filtered_list.append(intf)
        intf_list = filtered_list
        n2 = len(intf_list)
        logging.info('filtered list has {} nonz'.format(n2))
        logging.info('filtered by nonz threshold: removed {} intfs'.format(n1-n2))

    if args.partition_mode == 'random_by_patch':
        logging.info('Creating Dataset: Randlomly partitioning by patches')


        if args.nonoverlap_tr_tst:
            train_set = SubsiDataset(args,image_dir,mask_dir,intf_list,dset='train')
            test_set = SubsiDataset(args,image_dir,mask_dir,intf_list,dset='test')
            val_set = test_set
            n_train = len(train_set)
            n_val = len(val_set)
            n_test = len(test_set)
        else:
            dataset = SubsiDataset(args,image_dir,mask_dir,intf_list)
            n_total = int(len(dataset))
            n_val = int(n_total * val_percent)
            n_test = int(n_total * test_percent)
            n_train = n_total - n_val - n_test
            train_set, temp_set = random_split(dataset, [n_train, n_total - n_train], generator=torch.Generator().manual_seed(0))
            val_set, test_set = random_split(temp_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))

        logging.info('train val and test sets have {}, {}, {} samples'.format(len(train_set), len(val_set), len(test_set)) )
        buffer = io.BytesIO()
        pickle.dump(test_set, buffer)

        # Get the size in bytes
        size_in_bytes = buffer.tell()

        # Convert bytes to gigabytes
        size_in_gb = size_in_bytes / (1024 ** 3)

        logging.info(f"Estimated size of test_Dataset: {size_in_gb:.2f} GB")


        with open(outpath+'test_dataset_'+args.job_name+'.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    elif args.partition_mode == 'random_by_intf':
        logging.info('Creating Dataset: Randomly partitioning by Interferograms !!!')
        unique_intf_list = intf_list
        prev_dict=None
        if args.add_temporal or True:
            with open(args.intf_dict_path, "r") as f:
                intf_info = json.load(f)
            prev_dict,updated_intf_list = find_11day_sequences(intf_info,k_prev=args.k_prevs, restrict_to=unique_intf_list)
            unique_intf_list = updated_intf_list

        if args.test_data_to_exclude is not None:
            with open(args.test_data_to_exclude, 'rb') as file:
                test_data = pickle.load(file)
            test_list = test_data.ids
        else:
            test_list = None

        if args.retrain_with_fpz: #excluding previous run testlist intfs
            if test_list is None:
                logging.info('retrain mode has to have preset test list  !!')
                sys.exit(0)


            logging.info(f"excluding pre-determined test list: {test_list}")
            tv_list = list(set(unique_intf_list) - set(test_list))
            random.shuffle(tv_list)
            n_val = int(len(tv_list)*(val_percent))
            n_train = len(tv_list) - n_val
            train_list = tv_list[:n_train]
            val_list = tv_list[n_train:]

        else:
            if test_list is None:
                random.shuffle(unique_intf_list)
                n_val = int(len(unique_intf_list)*(val_percent))
                n_test = int(len(unique_intf_list)*(test_percent))
                n_train = len(unique_intf_list) - n_val - n_test
                if n_val == 0:
                    logging.info('not enough data for partitioning by interferograms !!')
                    sys.exit(0)
                train_list = unique_intf_list[:n_train]
                val_list = unique_intf_list[n_train:n_train+n_val]
                test_list = unique_intf_list[n_train+n_val:]

            else:
                tv_list = list(set(unique_intf_list) - set(test_list))
                logging.info(f' train + val sets include {len(tv_list)} intfs')
                logging.info(f' val percent is  {val_percent}')
                random.shuffle(tv_list)
                n_val = int(len(tv_list) * (val_percent))
                n_train = len(tv_list) - n_val
                if n_val == 0:
                    logging.info('not enough data for partitioning by interferograms !!')
                    sys.exit(0)

                train_list = tv_list[:n_train]
                val_list = tv_list[n_train:]
            preset_test_val_21 = True
            if preset_test_val_21:
                val_list = ['20210407_20210418','20210418_20210429','20210304_20210315','20210326_20210406']
                test_list = ['20210806_20210817','20210623_20210704','20210622_20210703','20210714_20210725']


                train_list = list(set(unique_intf_list) - set(test_list)-set(val_list))
                tmp_train_list = []
                for intf in train_list:
                    if intf[:4] != '2021':
                        tmp_train_list.append(intf)
                train_list = tmp_train_list

                logging.info(f' test interferograms: {test_list}')
                logging.info(f'val interferograms: {val_list}')
                logging.info(f'train interferograms: {train_list}')
        #
        train_list = ['20191129_20191210']
        test_list = ['20191129_20191210']
        val_list = ['20191129_20191210']

        train_set = SubsiDataset(args,image_dir,mask_dir,intrfrgrm_list=train_list,dset = 'train',seq_dict=prev_dict)
        val_set = SubsiDataset(args,image_dir,mask_dir,intrfrgrm_list=val_list,dset = 'val',seq_dict=prev_dict)
        test_set = SubsiDataset(args,image_dir, mask_dir, intrfrgrm_list=test_list,dset = 'test',seq_dict=prev_dict)
        logging.info('train intfs: ' + str(train_set.ids))
        logging.info('val intfs: ' + str(val_set.ids))
        logging.info('test intfs:' + str(test_set.ids))
        logging.info('num of intfs in train: ' + str(len(train_set.ids)))
        logging.info('num of intfs in val: ' + str(len(val_set.ids)))
        logging.info('num of intfs in test: ' + str(len(test_set.ids)))
        if (set(train_set.ids)).isdisjoint(set(val_set.ids)):
            logging.info( 'no commom train val intfs')
        if (set(train_set.ids)).isdisjoint(set(test_set.ids)):
            logging.info('no commom train test intfs')

### uncomment back!!!
        # assert set(train_set.ids).isdisjoint(set(val_set.ids)) and set(train_set.ids).isdisjoint(set(test_set.ids)) and set(val_set.ids).isdisjoint(
        #     set(test_set.ids)), 'there are common intfs in lists!'

        logging.info('train val and test sets have {}, {}, {} samples'.format(len(train_set), len(val_set), len(test_set)))
        buffer = io.BytesIO()
        pickle.dump(test_set, buffer)

        # Get the size in bytes
        size_in_bytes = buffer.tell()

        # Convert bytes to gigabytes
        size_in_gb = size_in_bytes / (1024 ** 3)

        logging.info(f"Estimated size of test_Dataset: {size_in_gb:.2f} GB")

        with open(outpath+'test_dataset_'+args.job_name+'.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    elif args.partition_mode == 'spatial':
        logging.info('Creating Dataset: by spatial partitioning !!!')
        train_set = SubsiDataset(args,image_dir,mask_dir,intf_list, dset = 'train',augment = args.augment)
        valtmp_set = SubsiDataset(args,image_dir,mask_dir,intf_list, dset = 'val')
        n_train = len(train_set)
        n_valtmp = len(valtmp_set)
        n_val = n_valtmp//2
        n_test = n_valtmp - n_val
        val_set,test_set = random_split(valtmp_set, [n_val,n_test])
        # val_indices = list(range(n_val))  # First n_val samples → validation
        # test_indices = list(range(n_val, n_valtmp))  # The rest → test
        #
        # # Define subsets
        # val_set = Subset(valtmp_set, val_indices)
        # test_set = Subset(valtmp_set, test_indices)

        logging.info('train val and test sets have {}, {}, {} samples'.format(len(train_set), len(val_set), len(test_set)) )
        logging.info('Spatial partitioning: Val percent is ' + str(int(100*n_val/(n_train+n_val+n_test))) + '%')
        logging.info('Spatial partitioning: test percent is ' + str(int(100*n_test/(n_train+n_val+n_test))) + '%')
        with open(outpath+'test_dataset_'+args.job_name+'.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    elif args.partition_mode == 'preset_by_intf':
        logging.info('Creating Dataset: preset partition by intf \n partiotion file: '+args.partition_file)
        with open(args.partition_file, 'r') as file:
            loaded_data = json.load(file)
        train_list = loaded_data['train']
        val_list = loaded_data['val']
        train_set = SubsiDataset(args, image_dir, mask_dir, intrfrgrm_list=train_list)
        val_set = SubsiDataset(args, image_dir, mask_dir, intrfrgrm_list=val_list)
        n_train = len(train_set)
        n_val = len(val_set)

    #     train_set, val_set = get_preset_partition()
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))
    global_step = 0

    #Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with (tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar):
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                # images: (B, 2T, H, W) if args.treat_nodata_regions else (B, T, H, W)
                assert images.shape[1] == model.n_channels, \
                    f'Network has {model.n_channels} input channels, but loaded images have {images.shape[1]}.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Split data and per-time validity when enabled
                if args.treat_nodata_regions:
                    T = images.shape[1] // 2
                    # x = images[:, :T, ...]     # not needed explicitly here
                    V = images[:, T:, ...]  # (B, T, H, W), 1=valid, 0=nodata
                    V_any = V.max(dim=1, keepdim=True).values  # (B, 1, H, W)
                else:
                    V_any = None  # not used

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits = model(images)  # (B,1,H,W) for binary; (B,C,H,W) for multiclass

                if model.n_classes == 1:
                    # Binary segmentation
                    y_float = true_masks.float().unsqueeze(1)  # (B,1,H,W)

                    if args.treat_nodata_regions:
                        # Masked BCE + masked Dice using V_any
                        loss_bce = masked_bce_with_logits(logits, y_float, V_any,pos_w=8)
                        loss_dice = masked_dice_loss_binary(logits, y_float, V_any)
                        # Optional tiny regularizer: discourage positives where ALL slices are nodata
                        no_data = (1.0 - V_any)
                        loss = loss_bce + loss_dice + 1e-3 * (torch.sigmoid(logits) * no_data).mean()
                    else:
                        # Unmasked baseline losses
                        loss = criterion(logits.squeeze(1), y_float.squeeze(1))
                        loss += dice_loss(torch.sigmoid(logits.squeeze(1)), y_float.squeeze(1), multiclass=False)

                else:
                    # Multiclass segmentation
                    if args.treat_nodata_regions:
                        # Expect masked CE + Soft Dice to broadcast V_any (B,1,H,W) across classes
                        loss = masked_ce_plus_softdice_multiclass(logits, true_masks, V_any)
                    else:
                        loss = criterion(logits, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                logging.info( '\n step: {}\n epoch: {}\n train loss: {:.8f} \n '.format(global_step,epoch,loss.item()))

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        Path(dir_validation).mkdir(parents=True, exist_ok=True)
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)


        val_score = evaluate(model, val_loader, device, amp, is_local=is_running_locally,out_path=str(dir_validation),epoch = epoch,save_val = args.save_val)
        scheduler.step(val_score)

        logging.info('Epoch {} Validation Dice score: {}'.format(epoch,val_score))
        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.dataset.mask_values if args.partition_mode == 'random_by_patch' and not args.nonoverlap_tr_tst else train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / (args.job_name + 'checkpoint_epoch{}.pth'.format(epoch))))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--test', dest='test', type=float, default=10.0,
                        help='Percent of the data that is used as test (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--stride',  type = int, default=2, help='train on patchs with given strides per window')

    parser.add_argument('--nonz_only', type = str, default='True', help='train only on non zero mask patches')

    parser.add_argument('--add_nulls_to_train', type = str, default='False', help='add patches with 0 sharp values')

    parser.add_argument('--patches_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='path to patches')
    parser.add_argument('--partition_mode', type=str, default='random_by_patch', choices=['random_by_patch', 'random_by_intf','spatial','preset_by_intf'], help='partition mode')
    parser.add_argument('--partition_file', type=str, default='partition_20_05_13h45.json', help=('preset partition file'))

    parser.add_argument('--train_on_11d_diff', type = str, default='True', help='train only on non zero mask patches')
    parser.add_argument('--job_name', type = str, default='', help='job name to add to output files')
    parser.add_argument('--intf_dict_path', type=str, default='./intf_coord.json', help='path to interferograms coord dict')
    parser.add_argument('--thresh_lat', type=float, default=31.4)
    parser.add_argument('--train_with_nonz_th', type = str, default='False', help='train only on non zero mask patches')
    parser.add_argument('--nonz_th',  nargs = '+', type = int, default=[350,150], help='north, south')
    parser.add_argument('--save_val', type = str, default='False', help='train only on non zero mask patches')
    parser.add_argument('--nonoverlap_tr_tst', type = str, default='False', help='')
    parser.add_argument('--retrain_with_fpz', type = str, default='False', help='')
    parser.add_argument('--test_data_to_exclude',type = str, default = None, help='path to previous run test dataset ')
    parser.add_argument('--use_cleaned_patches', action='store_true')
    parser.add_argument('--add_attn', action='store_true')
    parser.add_argument('--attn_unet', action='store_true')
    parser.add_argument('--augment', action='store_true',help='augment: only for spatial')
    parser.add_argument('--add_temporal', action='store_true')
    parser.add_argument('--k_prevs',type=int, default=2)
    parser.add_argument('--treat_nodata_regions', action='store_true')




    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # Configure the logging system
    logging.basicConfig(level=logging.INFO)  # Set the logging level (e.g., INFO)
    now = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    args.job_name = args.job_name + '_' + now
    outpath = './outputs/' + args.job_name + '/'
    os.makedirs(outpath, exist_ok=True)
    # Create a FileHandler and specify the log file name

    log_file = outpath + args.job_name + '_' + now + '.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)
    logging.info("train job started")
    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)

    if is_running_locally:
        logging.info("Running locally")
    else:
        logging.info("Running on a remote server")


    args.train_on_11d_diff = str2bool(args.train_on_11d_diff)
    args.nonz_only = str2bool(args.nonz_only)
    args.train_with_nonz_th = str2bool(args.train_with_nonz_th)
    args.add_nulls_to_train = str2bool(args.add_nulls_to_train)
    args.save_val = str2bool(args.save_val)
    args.nonoverlap_tr_tst = str2bool(args.nonoverlap_tr_tst)
    args.retrain_with_fpz = str2bool(args.retrain_with_fpz)

    dir_checkpoint = Path(outpath + 'checkpoints/')
    dir_validation = Path(outpath + 'validation/')
    dir_test = Path(outpath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = get_default_device()
    logging.info(f'Using device {device}')
    if args.add_temporal:
        num_c = args.k_prevs+1
    else:
        num_c=1
    if args.treat_nodata_regions:
        num_c = num_c*2

    if args.attn_unet:
        model = AttentionUNet(n_channels=num_c, n_classes=args.classes,bilinear=args.bilinear)
    else:
        model = UNet(n_channels=num_c, n_classes=args.classes, bilinear=args.bilinear,add_attn=args.add_attn)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device=device)
    try:
        train_model(
            args,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            test_percent = args.test / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
