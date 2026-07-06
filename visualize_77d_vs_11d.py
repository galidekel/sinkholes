import glob, json, os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

INTF_DIR   = "/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data"
JOB_11D    = "/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/job_07_01_01h53"
PRED_DIR   = "/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/job_07_05_16h18"
SHP_11D    = glob.glob(os.path.join(JOB_11D, "*combined.shp"))[0]
SHP_77D    = glob.glob(os.path.join(PRED_DIR, "*combined.shp"))[0]
COORD_JSON = "/home/labs/rudich/Rudich_Collaboration/sinkholes/intf_coord.json"

KEY_11 = "20220107_20220118"
KEY_77 = "20220107_20220325"

with open(COORD_JSON) as f:
    intf_info = json.load(f)

gdf_11d = gpd.read_file(SHP_11D)
gdf_77d = gpd.read_file(SHP_77D)
if gdf_11d.crs and gdf_11d.crs.to_epsg() != 4326:
    gdf_11d = gdf_11d.to_crs('EPSG:4326')
if gdf_77d.crs and gdf_77d.crs.to_epsg() != 4326:
    gdf_77d = gdf_77d.to_crs('EPSG:4326')

# ── load 11-day interferogram ─────────────────────────────────────────────────
info = intf_info[KEY_11]
north, east = info["north"], info["east"]
dx, dy      = info["dx"], info["dy"]
nlines, ncells = info["nlines"], info["ncells"]
bo = info.get("byte_order", "LSBFirst")

unw = glob.glob(os.path.join(INTF_DIR, f"tgeo_int_{KEY_11[:8]}*_{KEY_11[9:]}*.unw"))[0]
raw  = np.memmap(unw, dtype='>f4' if bo == 'MSBFirst' else '<f4', mode='r', shape=(nlines, ncells))
# crop to 77d footprint (same origin and dx/dy, just fewer rows/cols)
conf = np.load(os.path.join(PRED_DIR, f"{KEY_77}_pred.npy"))
nlines77, ncells77 = conf.shape
data = raw[:nlines77, :ncells77].copy()
extent77 = [east, east + dx * ncells77, north - dy * nlines77, north]
p2p = np.nanpercentile(data[data != 0], [2, 98])

# conf and extent77 already computed above when cropping 11d data

print(f"11d intf shape: {data.shape}  extent: {extent}")
print(f"77d conf shape: {conf.shape}  extent: {extent77}")
print(f"conf min={conf.min():.4f} max={conf.max():.4f} mean={conf.mean():.4f}")
print(f"11d polygons: {len(p11 := gdf_11d[gdf_11d['intf_key']==KEY_11])}")
print(f"77d polygons: {len(p77 := gdf_77d[gdf_77d['intf_key']==KEY_77])}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

ax1.imshow(data, extent=extent, cmap='jet', vmin=p2p[0], vmax=p2p[1], aspect='auto', origin='upper')
p11 = gdf_11d[gdf_11d['intf_key'] == KEY_11]
p77 = gdf_77d[gdf_77d['intf_key'] == KEY_77]
if not p11.empty:
    p11.boundary.plot(ax=ax1, color='white', linewidth=1.5, zorder=5)
ax1.set_aspect('auto')
ax1.set_xlim(extent77[0], extent77[1])
ax1.set_ylim(extent77[2], extent77[3])
ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax1.set_title(f"11-day: {KEY_11}", fontsize=12)
ax1.set_xlabel("Lon"); ax1.set_ylabel("Lat")

conf_nz = conf[conf > 0]
vmax77 = float(np.percentile(conf_nz, 98)) if conf_nz.size else 1.0
ax2.imshow(conf, extent=extent77, cmap='hot', vmin=0, vmax=vmax77, aspect='auto', origin='upper')
if not p77.empty:
    p77.boundary.plot(ax=ax2, color='white', linewidth=1.5, zorder=5)
ax2.set_aspect('auto')
ax2.set_xlim(extent77[0], extent77[1])
ax2.set_ylim(extent77[2], extent77[3])
ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax2.set_title(f"77-day confidence: {KEY_77}", fontsize=12)
ax2.set_xlabel("Lon")

plt.tight_layout()
plt.savefig("viz_77d_vs_11d.png", dpi=150, bbox_inches='tight')
plt.show()
print("saved viz_77d_vs_11d.png")
