import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluate import evaluate
from unet import *
import logging

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
    parser.add_argument('--test_intfs', type=str, default='False')

    return parser.parse_args()


    return parser.parse_args()
args = get_args()
test_intfs = str2bool(args.test_intfs)

with open(args.test_data_path, 'rb') as file:
    test_data = pickle.load(file)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)


if hasattr(test_data, 'ids') and test_intfs:
    test_intfs = test_data.ids
    test_image_data = test_data.image_data
    test_mask_data = test_data.mask_data
    logging.info('test_intfs: {}'.format(test_intfs))


net = UNet(n_channels=1, n_classes=1, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Loading model {args.model}')
logging.info(f'Using device {device}')

net.to(device=device)
state_dict = torch.load(args.model, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
net.eval()
logging.info('Model loaded!')

test_dir = 'test_data/'

test_dice_score =  evaluate(net, test_loader, device, amp=False, is_local=True, out_path=test_dir,epoch = 1,mode = 'test')
logging.info(f'test dice score is: {test_dice_score}')




