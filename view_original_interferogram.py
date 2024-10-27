from get_intf_info import *
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

intf_file_list = [file for file in listdir('./') if 'int_20211204' in file and '.ers' not in file]
gdf = gpd.read_file('sub_20231001.shp')
lidar_mask_df = gpd.read_file('lidar_mask_polygs.shp')
data_list,subset_polygs_list,extents,mask_polygs,start_dates,end_dates= [],[],[],[],[],[]

for intf_file in intf_file_list:
    mfile = intf_file + '.ers'


    intf = intf_file[9:17] + '_' + intf_file[25:33]
    intf_info = get_intf_coords(intf)
    x0 ,y0, dx, dy, nx, ny,lidar_mask = intf_info[0], intf_info[1], intf_info[2], intf_info[3], intf_info[4], intf_info[5], intf_info[8]
    mask_polyg = lidar_mask_df[lidar_mask_df['source'] == lidar_mask]
    mask_polygs.append(mask_polyg)
    xn = x0 + dx*nx
    yn = y0 - dy*ny
    extent = [x0, xn, yn, y0]
    start_date = intf[:4] + '-' + intf[4:6] + '-' + intf[6:8]
    end_date = intf[9:13]+'-'+intf[13:15]+'-'+intf[15:17]
    start_dates.append(start_date)
    end_dates.append(end_date)
    data = np.fromfile(intf_file, dtype=np.float32, count=-1, sep='', offset=0).reshape(ny, nx)
    with open(mfile) as f:
        for line in f:
            if 'ByteOrder' in line:
                byte_order = line.strip().split()[-1]

    if byte_order == 'MSBFirst':
        data = data.byteswap().newbyteorder('<')
    subset_polygs = gdf[(gdf['start_date'] == start_date) & (gdf['end_date'] == end_date)]
    data_list.append(data)
    subset_polygs_list.append(subset_polygs)
    extents.append(extent)
plt.rcParams['backend'] = 'Qt5Agg'

fig, axes = plt.subplots(1,len(intf_file_list), sharex=True, sharey=True)
if len(intf_file_list) == 1:
    axes = [axes]
for i,ax in enumerate(axes):
    ax.imshow(data_list[i], extent=extents[i], cmap='grey', aspect='auto')

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(str(start_dates[i])+' - '+str(end_dates[i]),fontsize=10)
# cbar = fig.colorbar(a, ax=ax1,orientation='vertical')
    for j in range(len(subset_polygs_list[i])):
        coords_e = subset_polygs_list[i].iloc[j]['geometry'].exterior.coords
        coords_list = list(coords_e)
        x, y = zip(*coords_list)
        ax.plot(x, y,c = 'b')
    for m in range(len(mask_polygs[i])):
        coords_e = mask_polygs[i].iloc[m]['geometry'].exterior.coords
        coords_list = list(coords_e)
        x, y = zip(*coords_list)
        ax.plot(x, y,c = 'b')




    plt.tight_layout()
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])

fig.suptitle('Interferogram '+ start_date + ' - ' + end_date,y=0.90)
plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust rect to fit the suptitle

plt.show()





