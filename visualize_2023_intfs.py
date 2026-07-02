import numpy as np
import glob, json, os
import geopandas as gpd
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, FloatText, Button, HBox, VBox, Label
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

dd      = Dropdown(options=intfs_2023, description="Intf:")
lon_min = FloatText(value=35.35, description="Lon min:", step=0.001, layout={"width": "200px"})
lon_max = FloatText(value=35.45, description="Lon max:", step=0.001, layout={"width": "200px"})
lat_min = FloatText(value=31.38, description="Lat min:", step=0.001, layout={"width": "200px"})
lat_max = FloatText(value=31.47, description="Lat max:", step=0.001, layout={"width": "200px"})
btn       = Button(description="Apply region", button_style="primary")
btn_reset = Button(description="Reset region", button_style="warning")
btn_prev  = Button(description="◀ Prev", layout={"width": "80px"})
btn_next  = Button(description="Next ▶", layout={"width": "80px"})

DEFAULT_LON_MIN, DEFAULT_LON_MAX = 35.35, 35.45
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 31.38, 31.47

def draw(intf_key=None, x0_crop=None, x1_crop=None, y0_crop=None, y1_crop=None):
    ax.cla()
    if intf_key is None:
        intf_key = dd.value
    if x0_crop is None:
        x0_crop, x1_crop = lon_min.value, lon_max.value
        y0_crop, y1_crop = lat_min.value, lat_max.value

    info = intf_info.get(intf_key, {})
    north, east = info["north"], info["east"]
    dx, dy = info["dx"], info["dy"]
    nlines, ncells = info["nlines"], info["ncells"]
    bo = info.get("byte_order", "LSBFirst")

    unw = find_unw(intf_key)
    if unw is not None:
        col0 = max(0, int((x0_crop - east) / dx))
        col1 = min(ncells, int((x1_crop - east) / dx) + 1)
        row0 = max(0, int((north - y1_crop) / dy))
        row1 = min(nlines, int((north - y0_crop) / dy) + 1)
        data = np.memmap(unw, dtype=np.float32, mode="r", shape=(nlines, ncells))
        crop = data[row0:row1, col0:col1].copy()
        if bo == "MSBFirst":
            crop = crop.byteswap().newbyteorder("<")
        crop = (crop + np.pi) / (2 * np.pi)
        step = max(1, max((row1-row0), (col1-col0)) // 2000)
        crop_extent = [east + col0*dx, east + col1*dx, north - row1*dy, north - row0*dy]
        ax.imshow(crop[::step, ::step], cmap="jet", vmin=0, vmax=1,
                  extent=crop_extent, origin="upper", aspect="auto")

    polygs = gdf_all[gdf_all["intf_key"] == intf_key]
    polygs.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=1.5)
    ax.set_xlim(x0_crop, x1_crop)
    ax.set_ylim(y0_crop, y1_crop)
    track = polygs["track"].iloc[0] if not polygs.empty else "?"
    ax.set_title(intf_key + "  |  " + str(len(polygs)) + " polygons  |  " + track)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.canvas.draw_idle()

def on_intf_change(change):
    if change["name"] == "value":
        draw()

def on_btn_click(b):
    draw()

def on_prev(b):
    idx = intfs_2023.index(dd.value)
    dd.value = intfs_2023[(idx - 1) % len(intfs_2023)]

def on_next(b):
    idx = intfs_2023.index(dd.value)
    dd.value = intfs_2023[(idx + 1) % len(intfs_2023)]

def on_reset(b):
    info = intf_info.get(dd.value, {})
    north, east = info["north"], info["east"]
    dx, dy = info["dx"], info["dy"]
    nlines, ncells = info["nlines"], info["ncells"]
    x0, x1 = east, east + ncells * dx
    y0, y1 = north - nlines * dy, north
    lon_min.value, lon_max.value = x0, x1
    lat_min.value, lat_max.value = y0, y1
    draw(x0_crop=x0, x1_crop=x1, y0_crop=y0, y1_crop=y1)

dd.observe(on_intf_change)
btn.on_click(on_btn_click)
btn_prev.on_click(on_prev)
btn_next.on_click(on_next)
btn_reset.on_click(on_reset)

controls = VBox([
    HBox([btn_prev, dd, btn_next]),
    HBox([Label("Region:"), lon_min, lon_max, lat_min, lat_max, btn, btn_reset])
])
display(VBox([controls, fig.canvas]))
draw()
