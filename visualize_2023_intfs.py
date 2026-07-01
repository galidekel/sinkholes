import numpy as np
import glob, json, os
import geopandas as gpd
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, VBox
from IPython.display import display

INTF_DIR   = "/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data"
SHP_PATH   = glob.glob("/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/"
                       "job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/"
                       "job_07_01_*/job_train_temp*combined.shp")[0]
COORD_JSON = "/home/labs/rudich/Rudich_Collaboration/sinkholes/intf_coord.json"

with open(COORD_JSON) as f:
    intf_info = json.load(f)

gdf_all = gpd.read_file(SHP_PATH)
gdf_2023 = gdf_all[gdf_all["start_date"].str.startswith("2023")]
intfs_2023 = sorted(gdf_2023["intf_key"].unique())
print("Loaded " + str(len(gdf_all)) + " polygons, " + str(len(intfs_2023)) + " intfs in 2023")

def find_unw(intf_key):
    d1, d2 = intf_key[:8], intf_key[9:]
    files = glob.glob(os.path.join(INTF_DIR, "tgeo_int_" + d1 + "*_" + d2 + "*.unw"))
    return files[0] if files else None

fig, ax = plt.subplots(figsize=(10, 8))
plt.tight_layout()

def draw(intf_key):
    ax.cla()
    info = intf_info.get(intf_key, {})
    north, east = info["north"], info["east"]
    dx, dy = info["dx"], info["dy"]
    nlines, ncells = info["nlines"], info["ncells"]
    bo = info.get("byte_order", "LSBFirst")
    extent = [east, east + ncells*dx, north - nlines*dy, north]

    unw = find_unw(intf_key)
    if unw is not None:
        data = np.fromfile(unw, dtype=np.float32).reshape(nlines, ncells)
        if bo == "MSBFirst":
            data = data.byteswap().newbyteorder("<")
        data = (data + np.pi) / (2 * np.pi)
        ax.imshow(data, cmap="jet", vmin=0, vmax=1,
                  extent=extent, origin="upper", aspect="equal")

    polygs = gdf_all[gdf_all["intf_key"] == intf_key]
    polygs.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=1.5)

    if not polygs.empty:
        minx, miny, maxx, maxy = polygs.total_bounds
        mx = (maxx - minx) * 0.3 + 0.005
        my = (maxy - miny) * 0.3 + 0.005
        ax.set_xlim(minx - mx, maxx + mx)
        ax.set_ylim(miny - my, maxy + my)

    track = polygs["track"].iloc[0] if not polygs.empty else "?"
    ax.set_title(intf_key + "  |  " + str(len(polygs)) + " polygons  |  " + track)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.canvas.draw_idle()

dd = Dropdown(options=intfs_2023, description="Intf 2023:")

def on_change(change):
    if change["name"] == "value":
        draw(change["new"])

dd.observe(on_change)
display(VBox([dd, fig.canvas]))
draw(intfs_2023[0])
