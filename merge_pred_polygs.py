import glob, os, json
import geopandas as gpd
import pandas as pd

POLYG_DIR_2022 = '/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/job_05_20_17h44/polygs'
PRED_2019_2021 = 'sub_pred_2019_2021.shp'
OUT_SHP        = 'sub_pred_2019_2022.shp'

with open('intf_coord.json') as f:
    intf_info = json.load(f)

def frame_to_track(intf_key):
    info = intf_info.get(intf_key, {})
    frame = info.get('frame', '')
    if frame == 'North':
        return 'asc'
    elif frame == 'South':
        return 'desc'
    return None

# ── merge 2022 polygs ─────────────────────────────────────────────────────────
shp_files = sorted(glob.glob(os.path.join(POLYG_DIR_2022, '*.shp')))
print(f'Found {len(shp_files)} shapefiles in 2022 job')

gdfs = []
for shp in shp_files:
    gdf = gpd.read_file(shp)
    key = os.path.basename(shp)[:17]
    gdf['intf_key']   = key
    gdf['start_date'] = pd.to_datetime(key[:8], format='%Y%m%d').strftime('%Y%m%d')
    gdf['end_date']   = pd.to_datetime(key[9:], format='%Y%m%d').strftime('%Y%m%d')
    gdf['track']      = frame_to_track(key)
    gdfs.append(gdf)

pred_2022 = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
pred_2022 = pred_2022[pred_2022.geometry.notna() & ~pred_2022.geometry.is_empty]
print(f'2022 polygons: {len(pred_2022)}')
print(f'2022 date range: {pred_2022.start_date.min()} → {pred_2022.start_date.max()}')

# ── load 2019-2021 and combine ────────────────────────────────────────────────
pred_2019_2021 = gpd.read_file(PRED_2019_2021)
print(f'2019-2021 polygons: {len(pred_2019_2021)}')

merged = gpd.GeoDataFrame(
    pd.concat([pred_2019_2021, pred_2022], ignore_index=True),
    crs=pred_2019_2021.crs
)
merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty]
print(f'Total polygons: {len(merged)}')
print(f'Full date range: {merged.start_date.min()} → {merged.start_date.max()}')

merged.to_file(OUT_SHP)
print(f'Saved → {OUT_SHP}')
