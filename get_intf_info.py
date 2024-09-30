
import json
def get_intf_coords(intf_name):
    intf_dict_file = open('intf_coord.json', 'r')
    intf_coords = json.load(intf_dict_file)
    x0 = intf_coords[intf_name]['east']
    y0 = intf_coords[intf_name]['north']
    dx = intf_coords[intf_name]['dx']
    dy = intf_coords[intf_name]['dy']
    nlines = intf_coords[intf_name]['nlines']
    ncells = intf_coords[intf_name]['ncells']
    lidar_mask = intf_coords[intf_name]['lidar_mask']

    x4000 = x0 + 4000*dx
    x8500 = x4000 + 4500*dx

    return (x0, y0, dx, dy,ncells, nlines, x4000, x8500,lidar_mask)
def get_intf_lidar_mask(intf_name):
    with open('lidar_intf_mask.txt', 'r') as f:
        mask = 'no_mask'
        for line in f:
            if intf_name[:8] == line[8:16] and intf_name[9:17] == line[24:32]:
                mask = line[40:49]
            elif intf_name[:8] == line[8:16]:
                mask = line[40:49]


    return mask

if __name__ == '__main__':
    intf = '20190730_20190810'
    mask = get_intf_lidar_mask(intf)
    print(mask)
