import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluate import evaluate
from unet import *
import logging

from attn_unet import *


def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg





# Load the test dataset from a pickle file
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='test')


    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--aux_model',type=str,default= None)
    parser.add_argument('--test_intfs', type=str, default='False')
    parser.add_argument('--unet_attn', action='store_true')
    parser.add_argument('--add_attn',action='store_true')
    parser.add_argument('--th', type=float, default=0.7)
    parser.add_argument('--b', type=int, default=5)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--k_prevs', type=int, default=0)

    return parser.parse_args()


    return parser.parse_args()
args = get_args()
test_intfs = str2bool(args.test_intfs)

with open(args.test_data_path, 'rb') as file:
    test_data = pickle.load(file)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


if hasattr(test_data, 'ids') and test_intfs:
    test_intfs = test_data.ids
    test_image_data = test_data.image_data
    test_mask_data = test_data.mask_data
    logging.info('test_intfs: {}'.format(test_intfs))


net = UNet(n_channels=args.k_prevs+1, n_classes=1, bilinear=False,add_attn=args.add_attn)
if args.aux_model is not None:
    net1= UNet(n_channels=args.k_prevs+1, n_classes=1, bilinear=False,add_attn=args.add_attn)
if args.unet_attn:
    net = AttentionUNet(n_channels=args.k_prevs+1, n_classes=1, bilinear=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Loading model {args.model}')
logging.info(f'Using device {device}')

net.to(device=device)
state_dict = torch.load(args.model, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
net.eval()
if args.aux_model is not None:
    net1.to(device=device)
    state_dict = torch.load(args.aux_model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net1.load_state_dict(state_dict)
    net1.eval()




logging.info('Model loaded!')

test_dir = 'test_data/'

test_dice_score =  evaluate(net, test_loader, device, amp=False, is_local=True, out_path=test_dir,epoch = 1,mode = 'test',net_aux=net,th = args.th,buffer = args.b,plot=args.plot)
logging.info(f'test dice score is: {test_dice_score}')




