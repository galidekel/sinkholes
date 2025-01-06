#!/usr/bin/env python3

# A code for evaluating ML processing of interferograms after identifiying subsidence due to sinkholes
# By: Ran Novitsky Nof @ Geological Survey of Israel, 2024

import argparse
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import rasterio
import rasterio.warp
import rasterio.features
import cartopy.crs as ccrs
import geopandas as gpd
import fiona
from logging.handlers import TimedRotatingFileHandler
from alive_progress import alive_bar
from scalebar import scale_bar

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')

# Default Parameters
LOG_LEVEL = 'DEBUG'
PTHSUBSIDENCE = './subsidence.shp'  # subsidence path and shapefile database (subsidence_20241215.zip)
B = 2  # buffer size in pixels
ITH = 0.7  # Pecentage of overlap between predicted and mapped
RTH = 0.25  # percentage of predictions per pixel threshold.
log = logging.getLogger('SubsidenceML')
log.setLevel(LOG_LEVEL)  # set at default log level


def get_utm_epsg_code(lon, lat):
    """
    Given a longitude (lon) and latitude (lat), determine the appropriate 
    UTM zone and return the corresponding EPSG code.
    """
    # Calculate UTM zone from longitude
    zone = int((lon + 180) / 6) + 1
    # EPSG codes: 
    # Northern Hemisphere: EPSG:326 + zone (e.g., EPSG:32633 for zone 33N)
    # Southern Hemisphere: EPSG:327 + zone (e.g., EPSG:32733 for zone 33S)
    if lat >= 0:
        epsg = f"EPSG:326{zone:02d}"
    else:
        epsg = f"EPSG:327{zone:02d}"
    return epsg

# west, south, east, north = 35.381324, 31.404450, 35.406325, 31.419371
def from_raster(filename, readdata=False, west=None, east=None, south=None, north=None):
    with rasterio.open(filename) as dataset:
        if all([west, east, south, north]):
            window = rasterio.windows.from_bounds(west, south, east, north, dataset.transform)
            transform = dataset.window_transform(window)
        else:
            window = rasterio.windows.from_bounds(**dataset.bounds._asdict(), transform=dataset.transform)
            transform = dataset.transform
        nrows, ncols = int(window.height), int(window.width)
        dx, _, x0, _, dy, y0, *_ = list(transform)
        if readdata:
            log.debug('Reading dataset')
            data = dataset.read(1, window=window)
            nrows, ncols = data.shape  # just to make sure...
        else:
            data = None
    try:
        master, slave = os.path.split(filename)[1].split('.')[0].split('_')[2:4]
    except:
        master, slave = 'm', 's'
    return master, slave, nrows, ncols, x0, y0, dx, dy, data


def add_scalebar(ax, dist_km, units='km', km2unit=1, color='k', lw=3, location='upper right', pad=0.5, fontproperties={'weight':'bold'}):
    """Adds Scalebar to map
      ax: axis to add scalebar to. Should be a cartopy axis
      dist_km: Distance in km for the scale bar
      units: The name of the units (label near numbers)
      km2unit: The number on the scalebar will be dist_km * km2unit, for meters: dist_km=1, km2unit=1000.
      color: color for text and bar
      lw: line width for bar (3 is recommanded)
      location: where to locate the bar. See AnchoredSizeBar for applicable locations
      pad: Distance between bar and axis borders
      fontproperties: fornt properties dictionary
    Returns: AnchoredSizeBar object
    """
    center_lat = np.mean(ax.get_ylim())
    km_per_degree_lon = 111.320 * np.cos(center_lat * np.pi/180)
    dist_degrees = dist_km / km_per_degree_lon
    dist = dist_km * km2unit
    # Add scale bar in degrees
    scalebar = AnchoredSizeBar(ax.transData,
                                dist_degrees, f'{dist:g} {units}', location, 
                                sep=lw*2,
                                pad=pad,
                                color=color,
                                frameon=False,
                                fontproperties={fontproperties}
                                )
    scalebar.size_bar.get_children()[0].set_lw(lw)
    ax.add_artist(scalebar)
    return scalebar

class interferogram_score():
    def __init__(self, subsidence=PTHSUBSIDENCE, scores=None):
        self.subsidence = gpd.read_file(subsidence)
        if scores is None:
            self.scores = gpd.pd.DataFrame(names=['master', 'slave', 'Number_Mapped', 'Number_Predicted', 'TP', 'FN', 'FP', 'Recall', 'Precision', 'b', 'ITh'])
        else:
            self.scores = gpd.pd.read_csv(scores)

    def proc_intf(inteferogram):
        #interferogram = 'tgeo_mask_20241003T034857_20241116T034857.unw.ers'
        master, slave, nrows, ncols, x0, y0, dx, dy, data = from_raster(interferogram, readdata=False)
        mapped = self.get_mapped(master, slave)
        preds = self.get_preds(master, slave)


    def get_mapped(self, master, slave):
        mapped = self.subsidence.loc[(self.subsidence.start_date.dt.date==gpd.pd.to_datetime(master).date()) & (self.subsidence.end_date.dt.date==gpd.pd.to_datetime(slave).date())]
        return mapped
    
    def get_preds(self, master, slave):
        preds = gpd.read_file(f'tgeo_pred_{master}_{slave}.unw.shp')
        return preds
    
    def get_score(self, mapped, preds, buffersizem=15, ITh=0.7):
        #c = preds.unary_union.centroid
        c = preds.union_all().centroid
        utm = get_utm_epsg_code(c.x, c.y)
        # add buffer
        wgs = preds.crs
        buffered = gpd.GeoDataFrame([], geometry=preds.to_crs(utm).buffer(buffersizem).to_crs(wgs))
        # Get intersection area and it's percentage
        intersections = gpd.sjoin(mapped, buffered, how="inner", predicate="intersects")
        # Create sets to store indices of matched mapped polygons
        tp_mapped = set()
        #tp_predictions = set()
        for idx in set(intersections.index):
            mapped_poly = mapped.loc[idx].geometry
            preds_idx = intersections.loc[idx].index_right  # the index of the matched prediction polygon(s) from the join
            preds_poly = buffered.loc[preds_idx, 'geometry']
            if not type(preds_idx) is np.int64:
                preds_poly = preds_poly.union_all()
            
            # Compute intersection geometry
            intersect_geom = preds_poly.intersection(mapped_poly)
            if intersect_geom.is_empty:
                continue
            intersection_area = intersect_geom.area
            mapped_area = mapped_poly.area
            # Check if intersection meets the threshold
            if (intersection_area / mapped_area) >= ITh:
                #[tp_predictions.add(i) for i in preds_idx]  # store the index of the prediction polygon that qualifies
                tp_mapped.add(idx)  # recall mapped polygons matched

        # False Positives (FP): predictions not in TP
        #fp_predictions = set(buffered.index) - tp_predictions
        # False Negatives (FN): mapped polygons that have no prediction meeting threshold
        fn_mapped = set(mapped.index) - tp_mapped
        # Get polygons for TP, FP and FN:
        #TP = preds.loc[list(tp_predictions)]
        #FP = preds.loc[list(fp_predictions)]
        #FN = mapped.loc[list(fn_mapped)]
        GTS = mapped.loc[list(tp_mapped)]
        Recall = GTS.geometry.area.sum()/mapped.geometry.area.sum()
        Precision = GTS.geometry.area.sum()/preds.geometry.area.sum()
        return GTS, Recall, Precision

    def plot(self, interferogram, west, east, south, north, mapped, preds, GTS):
        master, slave, nrows, ncols, x0, y0, dx, dy, data = from_raster(interferogram, True, west, east, south, north)
        east = west + dx * ncols
        south = north + dy * nrows
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, color='None')
        im = ax.imshow(np.ma.masked_where(data==0, data),
          cmap='turbo', 
          extent=(west, east, south, north), 
          transform=ccrs.PlateCarree(), interpolation='nearest')
        im.cmap.set_bad('k')
        mapped.plot(ax=ax, facecolor='None', edgecolor='k', lw=3, label = 'Mapped')
        m = ax.collections[-1]
        preds.plot(ax=ax, facecolor='None', edgecolor='w', lw=2, label = 'Predictions')
        p = ax.collections[-1]
        cb = plt.colorbar(im, label='LOS phase [radians]', shrink=0.5)
        cb.set_ticks([0,0.5,1])
        cb.set_ticklabels([r'$-\pi$','0',r'$\pi$'])
        xt = ax.get_xticks()
        yt = ax.get_yticks()
        ax.set_xticks(xt, crs=ccrs.PlateCarree())
        ax.set_yticks(yt, crs=ccrs.PlateCarree())
        gl.remove()
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        sb = add_scalebar(ax, 0.5, color='w')
        