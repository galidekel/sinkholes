import argparse
import logging
import os
import random
import sys
import torch
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

# import wandb
from evaluate import evaluate
from unet import UNet
from sinkholes_data_loading import *
import logging
from dice_score import dice_loss

from datetime import datetime



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
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0

):
    # 1. Create dataset
    patch_size = tuple(args.patch_size)
    H, W = patch_size

    image_dir = args.patches_dir + 'data_patches_H' + str(H) + '_W' + str(W) +'_strpp'+str(args.stride) + ('_11days' if args.train_on_11d_diff else '')
    mask_dir = args.patches_dir + 'mask_patches_H' + str(H) + '_W' + str(W)  +'_strpp'+str(args.stride) + ('_11days' if args.train_on_11d_diff else '')
    #

        #sys.exit(0)
    logging.info('input patch directories: {} and {}'.format(image_dir, mask_dir))
    assert os.path.exists(image_dir) and os.path.exists(mask_dir), 'The data you are requesting does not exist, please check if you prepared it at the preparation stage'
    if args.partition_mode == 'random_by_patch':
        logging.info('Creating Dataset: Randlomly partitioning by patches')
        dataset = SubsiDataset(args,image_dir,mask_dir)
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif args.partition_mode == 'random_by_intf':
        logging.info('Creating Dataset: Randlomly partitioning by Interferograms !!!')
        intf_list = listdir(image_dir)
        unique_intf_list = [file for file in intf_list if 'nonz' in (file)] if args.nonz_only else [file for file in intf_list if 'nonz' not in (file)]
        random.shuffle(unique_intf_list)
        n_val = int(len(unique_intf_list)*(val_percent))
        n_train = len(unique_intf_list) - n_val
        if n_val == 0:
            logging.info('not enough data for partitioning by interferograms !!')
            sys.exit(0)
        train_list = unique_intf_list[:n_train]
        val_list = unique_intf_list[n_train:]
        train_set = SubsiDataset(args,image_dir,mask_dir,intrfrgrm_list=train_list)
        val_set = SubsiDataset(args,image_dir,mask_dir,intrfrgrm_list=val_list)
    elif args.partition_mode == 'spatial':
        logging.info('Creating Dataset: by spatial partitioning !!!')
        train_set = SubsiDataset(args,image_dir,mask_dir, dset = 'train')
        val_set = SubsiDataset(args,image_dir,mask_dir, dset = 'val')
        n_train = len(train_set)
        n_val = len(val_set)
        logging.info('Spatial partitioning: Val percent is ' + str(int(100*n_val/(n_train+n_val))) + '%')
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
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

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

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with (tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar):
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if is_running_locally:
                        images_np =  images.detach().numpy()
                        masks_pred_np = masks_pred.detach().numpy()
                        true_masks_np = true_masks.detach().numpy()
                        true_masks_np = np.expand_dims(true_masks_np,axis = 1)
                        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
                        ax1.imshow(images_np[0,0,:,:])
                        ax2.imshow(masks_pred_np[0,0,:,:])
                        ax3.imshow(true_masks_np[0,0,:,:])
                        plt.show()

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                logging.info( '\n step: {}\n epoch: {}\n train loss: {:.8f} \n '.format(global_step,epoch,loss.item()))
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        Path(dir_validation).mkdir(parents=True, exist_ok=True)
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)


        val_score = evaluate(model, val_loader, device, amp, is_local=is_running_locally,out_path=str(dir_validation),epoch = epoch)
        scheduler.step(val_score)

        logging.info('Epoch {} Validation Dice score: {}'.format(epoch,val_score))
        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.dataset.mask_values if args.partition_mode == 'random_by_patch' else train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / (args.job_name + 'checkpoint_epoch{}.pth'.format(epoch))))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--stride',  type = int, default=2, help='train on patchs with given strides per window')

    parser.add_argument('--nonz_only', type = str, default='True', help='train only on non zero mask patches')
    parser.add_argument('--patches_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='path to patches')
    parser.add_argument('--partition_mode', type=str, default='random_by_patch', choices=['random_by_patch', 'random_by_intf','spatial','preset_by_intf'], help='partition mode')
    parser.add_argument('--partition_file', type=str, default='partition_20_05_13h45.json', help=('preset partition file'))

    parser.add_argument('--train_on_11d_diff', type = str, default='True', help='train only on non zero mask patches')
    parser.add_argument('--job_name', type = str, default='', help='job name to add to output files')
    parser.add_argument('--intf_dict_path', type=str, default='./intf_coord.json', help='path to interferograms coord dict')
    parser.add_argument('--thresh_lat', type=float, default=31.4)


    return parser.parse_args()


if __name__ == '__main__':
    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)

    if is_running_locally:
        print("Running locally")
    else:
        print("Running on a remote server")

    args = get_args()
    args.train_on_11d_diff = str2bool(args.train_on_11d_diff)
    args.nonz_only = str2bool(args.nonz_only)
    # Configure the logging system
    logging.basicConfig(level=logging.INFO)  # Set the logging level (e.g., INFO)
    now = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    args.job_name = args.job_name + '_' + now
    outpath = './outputs/' + args.job_name + '/'
    os.makedirs(outpath, exist_ok=True)
    # Create a FileHandler and specify the log file name

    log_file =  outpath + args.job_name + '_' + now +'.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)

    # Get the current date and time
    current_time = datetime.now()

    # Format the current time without seconds
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

    dir_checkpoint = Path(outpath + 'checkpoints/')
    dir_validation = Path(outpath + 'validation/')



    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = get_default_device()
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=1 for intrfrgrm images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

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
