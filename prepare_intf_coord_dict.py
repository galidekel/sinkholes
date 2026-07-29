from os import listdir
import argparse
import json
from get_intf_info import *
import logging


parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')

parser.add_argument('--intf_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/')
parser.add_argument('--out_dir', type=str, default='./')

args = parser.parse_args()

# frame threshold: same latitude cutoff previously used by the separate "add_frame to dict.py"
# step -- computed inline here instead so this script alone produces a complete, patchify-ready
# entry (prepare_intrfrgrm_pathches.py hard-requires 'frame') in one pass.
FRAME_THRESH = 31.6

# merge into any existing intf_coord.json instead of overwriting it -- this script is commonly
# re-run to add a new batch of .ers files, and previously it wiped 'frame'/'nonz_num' for every
# already-processed interferogram, not just the new ones
out_path = args.out_dir + 'intf_coord.json'
try:
    with open(out_path, 'r') as f:
        intf_dict = json.load(f)
except FileNotFoundError:
    intf_dict = dict()

for file in listdir(args.intf_dir):
    if file.endswith('.ers'):
        print(file)
        with open(args.intf_dir + file, 'r') as f:
            for line in f:
                for line in f:
                    if 'NrOfLines' in line:
                        NLINES = int(line.strip().split()[-1])
                    if 'NrOfCellsPerLine' in line:
                        NCELLS = int(line.strip().split()[-1])
                    if 'Northings' in line:
                        y0 = float(line.strip().split()[-1])
                    if 'Eastings' in line:
                        x0 = float(line.strip().split()[-1])
                    if 'Ydimension' in line:
                        dY = float(line.strip().split()[-1])
                    if 'Xdimension' in line:
                        dX = float(line.strip().split()[-1])
                    if 'ByteOrder' in line:
                        byte_order = line.strip().split()[-1]
        intfrgrm_name = file.split('.')[0][9:17] + file.split('.')[0][24:33]
        intf_lidar_mask = get_intf_lidar_mask(intfrgrm_name)
        print(intf_lidar_mask)
        if intf_lidar_mask == 'no_mask':
            logging.info('Note: No LiDAR mask for {}. Please decide what todo with it.'.format(intfrgrm_name))
        frame = 'North' if y0 > FRAME_THRESH else 'South'
        # preserve any existing 'nonz_num' for this intf (set later by check_patches.py) rather
        # than clobbering it back to missing on a re-run
        nonz_num = intf_dict.get(intfrgrm_name, {}).get('nonz_num', 'none')
        intf_dict[intfrgrm_name] = {'north': y0,'east':x0, 'nlines': NLINES, 'ncells':NCELLS, 'dy' : dY, 'dx':dX,'byte_order':byte_order, 'lidar_mask':intf_lidar_mask, 'frame':frame, 'nonz_num':nonz_num}


with open(out_path, 'w') as json_file:
    json.dump(intf_dict, json_file, indent=4)###