import glob, json, os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, Button, HBox, VBox
from IPython.display import display

INTF_DIR    = "/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data"
JOB_11D     = "/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/job_07_01_01h53"
PRED_DIR    = "/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/job_07_05_16h18"
SHP_11D     = glob.glob(os.path.join(JOB_11D, "*combined.shp"))[0]
SHP_77D     = glob.glob(os.path.join(PRED_DIR, "*combined.shp"))[0]
COORD_JSON  = "/home/labs/rudich/Rudich_Collaboration/sinkholes/intf_coord.json"

with open(COORD_JSON) as f:
    intf_info = json.load(f)

gdf_11d = gpd.read_file(SHP_11D)
gdf_77d = gpd.read_file(SHP_77D)

# build list of (77d_key, 11d_key) pairs: same start date, 11-day baseline from intf_info
pairs = []
for key77 in sorted(gdf_77d["intf_key"].unique()):
    start = key77[:8]
    key11 = next(
        (k for k, v in intf_info.items()
         if k[:8] == start and
         (pd.Timestamp(k[9:17]) - pd.Timestamp(k[:8])).days == 11),
        None
    )
    pairs.append((key77, key11))

pair_labels = [f"{k77}  (11d: {k11 or 'N/A'})" for k77, k11 in pairs]

def find_unw(intf_key):
    d1, d2 = intf_key[:8], intf_key[9:]
    files = glob.glob(os.path.join(INTF_DIR, f"tgeo_int_{d1}*_{d2}*.unw"))
    return files[0] if files else None

def load_intf(intf_key):
    info = intf_info.get(intf_key, {})
    north, east = info["north"], info["east"]
    dx, dy = info["dx"], info["dy"]
    nlines, ncells = info["nlines"], info["ncells"]
    bo = info.get("byte_order", "LSBFirst")
    unw = find_unw(intf_key)
    if unw is None:
        return None, north, east, dx, dy, nlines, ncells
    step = max(1, max(nlines, ncells) // 1000)
    raw = np.memmap(unw, dtype='>f4' if bo == 'MSBFirst' else '<f4', mode='r', shape=(nlines, ncells))
    data = raw[::step, ::step].copy()
    return data, north, east, dx, dy, nlines // step, ncells // step

_cache = {}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
plt.tight_layout()

dd       = Dropdown(options=pair_labels, description="Intf:")
dd.layout.width = '600px'
btn_prev = Button(description="◀ Prev", layout={"width": "80px"})
btn_next = Button(description="Next ▶", layout={"width": "80px"})

def draw(idx=None):
    if idx is None:
        idx = dd.index
    key77, key11 = pairs[idx]
    print(f"key77={key77}  key11={key11}")
    pred_path = os.path.join(PRED_DIR, f"{key77}_pred.npy")
    print(f"conf map exists: {os.path.exists(pred_path)}  path: {pred_path}")
    polygs_11 = gdf_11d[gdf_11d['intf_key'] == key11]
    print(f"11d polygons matching key11: {len(polygs_11)}")
    unw = find_unw(key11) if key11 else None
    print(f"unw file: {unw}")

    ax1.cla(); ax2.cla()

    # ── left panel: 11-day interferogram + 11-day predicted polygons ──────────
    if key11 and key11 not in _cache:
        _cache[key11] = load_intf(key11)
    if key11 and _cache[key11][0] is not None:
        data, north, east, dx, dy, nlines, ncells = _cache[key11]
        extent = [east, east + dx * ncells, north - dy * nlines, north]
        p2p = np.nanpercentile(data[data != 0], [2, 98]) if (data != 0).any() else (data.min(), data.max())
        ax1.imshow(data, extent=extent, cmap='jet', vmin=p2p[0], vmax=p2p[1], aspect='auto', origin='upper')
        polygs_11 = gdf_11d[gdf_11d['intf_key'] == key11]
        if not polygs_11.empty:
            polygs_11.boundary.plot(ax=ax1, color='white', linewidth=1)
        ax1.set_aspect('auto')
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax1.set_title(f"11-day: {key11}", fontsize=10)
        ax1.set_xlabel("Lon"); ax1.set_ylabel("Lat")
    else:
        ax1.text(0.5, 0.5, f"No 11-day intf\nfor {key77[:8]}", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f"11-day: {key11 or 'N/A'}", fontsize=10)

    # ── right panel: 77-day confidence map + 77-day predicted polygons ────────
    pred_path = os.path.join(PRED_DIR, f"{key77}_pred.npy")
    if os.path.exists(pred_path):
        conf = np.load(pred_path)
        info77 = intf_info.get(key77, {})
        north77, east77 = info77.get("north", 0), info77.get("east", 0)
        dx77, dy77 = info77.get("dx", 1), info77.get("dy", 1)
        nlines77, ncells77 = conf.shape
        extent77 = [east77, east77 + dx77 * ncells77, north77 - dy77 * nlines77, north77]
        ax2.imshow(conf, extent=extent77, cmap='hot', vmin=0, vmax=1, aspect='auto', origin='upper')
        polygs_77 = gdf_77d[gdf_77d['intf_key'] == key77]
        if not polygs_77.empty:
            polygs_77.boundary.plot(ax=ax2, color='cyan', linewidth=1)
        ax2.set_aspect('auto')
        ax2.set_xlim(extent77[0], extent77[1])
        ax2.set_ylim(extent77[2], extent77[3])
        ax2.set_title(f"77-day confidence: {key77}", fontsize=10)
        ax2.set_xlabel("Lon")
    else:
        ax2.text(0.5, 0.5, f"No confidence map\n{pred_path}", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f"77-day: {key77}", fontsize=10)

    fig.canvas.draw_idle()

def on_change(change):
    if change['name'] == 'index':
        draw(change['new'])

def on_prev(b):
    if dd.index > 0:
        dd.index -= 1

def on_next(b):
    if dd.index < len(pairs) - 1:
        dd.index += 1

dd.observe(on_change)
btn_prev.on_click(on_prev)
btn_next.on_click(on_next)

display(VBox([HBox([btn_prev, btn_next, dd]), fig.canvas]))
draw(0)
