#!/usr/bin/env python3
"""
Browser-based interferogram viewer.

Runs on the remote server next to the data; you open it in your local
browser through an SSH port-forward:

    (on the cluster)  python intf_viewer.py --polyg_path <predicted_polygs.shp or dir> [--port 8050]
    (on your laptop)  ssh -L 8050:localhost:8050 <cluster>
    then browse to    http://localhost:8050

Features:
  - full-resolution interferogram display with smooth zoom/pan (Leaflet)
  - predicted polygons overlaid (combined shapefile with intf_key/start_date
    columns, or a directory of per-intf *_predicted_polygs.shp)
  - step back/forth in time with arrow keys, North and South frames as
    separate chronological sequences (switching frames jumps to the
    nearest date)
  - the current zoom/center is kept while stepping, so you can stay locked
    on one area across time
  - neighbor interferograms are pre-rendered in the background for fast
    navigation
"""
import argparse
import io
import json
import logging
import os
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image

try:
    from matplotlib import colormaps as _colormaps   # matplotlib >= 3.6
    def get_cmap(name):
        return _colormaps[name]
except ImportError:                                    # older matplotlib
    from matplotlib import cm as _cm
    def get_cmap(name):
        return _cm.get_cmap(name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

Image.MAX_IMAGE_PIXELS = None  # interferograms are large; we trust our own files


# ----------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser(description='Browser-based interferogram viewer')
    p.add_argument('--input_dir', type=str,
                   default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/',
                   help='directory with the tgeo_int_*.unw files')
    p.add_argument('--intf_dict_path', type=str, default='./intf_coord.json',
                   help='per-interferogram coordinate dictionary')
    p.add_argument('--polyg_path', type=str, default=None,
                   help='predicted polygons: a combined shapefile, or a directory '
                        'of per-intf *_predicted_polygs.shp')
    p.add_argument('--days_diff', type=int, default=11,
                   help='only show interferograms with this duration in days')
    p.add_argument('--port', type=int, default=8050)
    p.add_argument('--cache_dir', type=str, default='./intf_viewer_cache',
                   help='where rendered images are cached')
    p.add_argument('--fmt', type=str, default='jpg', choices=['jpg', 'png'],
                   help='rendered image format (jpg = smaller/faster, png = lossless)')
    p.add_argument('--clip_pct', nargs=2, type=float, default=[2.0, 98.0],
                   help='percentile clip for display normalization')
    return p.parse_args()


ARGS = get_args()

# ----------------------------------------------------------------- index
with open(ARGS.intf_dict_path) as f:
    COORD = json.load(f)


def parse_key(key):
    s, e = key.split('_')
    return datetime.strptime(s, '%Y%m%d'), datetime.strptime(e, '%Y%m%d')


def build_sequences():
    seqs = {'North': [], 'South': []}
    for key, info in COORD.items():
        frame = info.get('frame')
        if frame not in seqs:
            continue
        try:
            sd, ed = parse_key(key)
        except ValueError:
            continue
        if (ed - sd).days != ARGS.days_diff:
            continue
        seqs[frame].append(key)
    for frame in seqs:
        seqs[frame].sort()
    return seqs


SEQS = build_sequences()
logging.info('11-day interferograms: %d North, %d South',
             len(SEQS['North']), len(SEQS['South']))

# map intf key -> .unw filename (filename carries times, key carries dates only)
UNW_BY_KEY = {}
for fn in os.listdir(ARGS.input_dir):
    if not fn.endswith('.unw'):
        continue
    m = re.match(r'tgeo_int_(\d{8})T\d+_(\d{8})T\d+\.unw$', fn)
    if m:
        UNW_BY_KEY[f'{m.group(1)}_{m.group(2)}'] = fn
logging.info('found %d .unw files in %s', len(UNW_BY_KEY), ARGS.input_dir)

# ----------------------------------------------------------------- polygons
POLYG_GDF = None       # combined-shapefile mode
POLYG_DIR = None       # per-intf-directory mode
def _to_lonlat(gdf):
    """Match view_intf_quick.py: display coordinates are EPSG:4326 lon/lat."""
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')
    return gdf


if ARGS.polyg_path:
    import geopandas as gpd
    if os.path.isdir(ARGS.polyg_path):
        POLYG_DIR = ARGS.polyg_path
        logging.info('polygons: per-intf directory %s', POLYG_DIR)
    else:
        POLYG_GDF = _to_lonlat(gpd.read_file(ARGS.polyg_path))
        logging.info('polygons: %d features from %s (crs: %s, columns: %s)',
                     len(POLYG_GDF), ARGS.polyg_path, POLYG_GDF.crs,
                     list(POLYG_GDF.columns))


def polygons_geojson(key):
    """GeoJSON FeatureCollection of the predicted polygons for one intf."""
    empty = '{"type":"FeatureCollection","features":[]}'
    if POLYG_GDF is not None:
        g = POLYG_GDF
        if 'intf_key' in g.columns:
            sel = g[g['intf_key'] == key]
        elif 'start_date' in g.columns and 'end_date' in g.columns:
            sd, ed = key.split('_')
            norm = lambda col: g[col].astype(str).str.replace('-', '', regex=False)
            sel = g[(norm('start_date') == sd) & (norm('end_date') == ed)]
        else:
            sel = g  # no date columns: show everything
        return sel.to_json() if len(sel) else empty
    if POLYG_DIR is not None:
        import geopandas as gpd
        hits = sorted(Path(POLYG_DIR).glob(f'*{key}*.shp'))
        if hits:
            return _to_lonlat(gpd.read_file(hits[0])).to_json()
    return empty


# ----------------------------------------------------------------- rendering
os.makedirs(ARGS.cache_dir, exist_ok=True)
_render_lock = threading.Lock()
_rendering = set()


def intf_extent(key):
    info = COORD[key]
    x0, y0 = info['east'], info['north']
    dx, dy = info['dx'], info['dy']
    nlines, ncells = info['nlines'], info['ncells']
    # [south, west, north, east] for Leaflet bounds
    return [y0 - dy * nlines, x0, y0, x0 + dx * ncells]


def render_image(key, cmap_name):
    """Render one .unw to a cached image file; return its path."""
    out = os.path.join(ARGS.cache_dir, f'{key}_{cmap_name}.{ARGS.fmt}')
    if os.path.exists(out):
        return out
    with _render_lock:
        if key + cmap_name in _rendering:
            # someone else is rendering it; wait by polling
            pass
        _rendering.add(key + cmap_name)
    try:
        if os.path.exists(out):
            return out
        info = COORD[key]
        fn = UNW_BY_KEY.get(key)
        if fn is None:
            raise FileNotFoundError(f'no .unw file for {key}')
        data = np.fromfile(os.path.join(ARGS.input_dir, fn),
                           dtype=np.float32).reshape(info['nlines'], info['ncells'])
        if info.get('byte_order') == 'MSBFirst':
            data = data.byteswap().newbyteorder('<')

        valid = data != 0
        if valid.any():
            lo, hi = np.percentile(data[valid], ARGS.clip_pct)
        else:
            lo, hi = 0.0, 1.0
        # quantize to 8-bit indices, then colorize through a 256-entry LUT —
        # avoids building huge float RGBA intermediates for large rasters
        idx = np.clip((data - lo) / max(hi - lo, 1e-12), 0, 1)
        idx = (idx * 255).astype(np.uint8)
        if cmap_name == 'gray':
            gray = np.where(valid, idx, 0).astype(np.uint8)
            img = Image.fromarray(gray, mode='L')
        else:
            lut = (get_cmap(cmap_name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            rgb = lut[idx]
            rgb[~valid] = 0  # no-data black
            img = Image.fromarray(rgb, mode='RGB')
        tmp = out + '.tmp'
        if ARGS.fmt == 'jpg':
            img.save(tmp, format='JPEG', quality=87)
        else:
            img.save(tmp, format='PNG')
        os.replace(tmp, out)
        logging.info('rendered %s (%s)', key, cmap_name)
        return out
    finally:
        with _render_lock:
            _rendering.discard(key + cmap_name)


def prerender_neighbors(key, cmap_name):
    """Warm the cache for the previous/next intf of the same frame."""
    for frame, keys in SEQS.items():
        if key in keys:
            i = keys.index(key)
            for j in (i - 1, i + 1):
                if 0 <= j < len(keys):
                    k = keys[j]
                    threading.Thread(target=render_image, args=(k, cmap_name),
                                     daemon=True).start()
            return


# ----------------------------------------------------------------- web page
PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"><title>Interferogram Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin:0; height:100%; }
  #map { position:absolute; top:44px; bottom:0; left:0; right:0; background:#111; }
  #bar { position:absolute; top:0; height:44px; left:0; right:0; background:#1e1e1e;
         color:#ddd; font:13px/44px sans-serif; padding:0 12px; box-sizing:border-box;
         display:flex; gap:14px; align-items:center; z-index:1000; }
  #bar b { color:#fff; }
  #bar button, #bar select { background:#333; color:#ddd; border:1px solid #555;
         border-radius:4px; padding:3px 10px; cursor:pointer; }
  #bar label { cursor:pointer; }
  .pix { image-rendering: pixelated; }
  #loading { color:#f90; display:none; }
</style>
</head>
<body>
<div id="bar">
  <span>
    <label><input type="radio" name="frame" value="North" checked> North</label>
    <label><input type="radio" name="frame" value="South"> South</label>
  </span>
  <button id="prev">&#8592; prev</button>
  <button id="next">next &#8594;</button>
  <b id="title">-</b>
  <span id="pos"></span>
  <select id="cmap">
    <option value="gray" selected>gray</option>
    <option value="jet">jet</option>
    <option value="twilight">twilight</option>
    <option value="RdBu">RdBu</option>
  </select>
  <label><input type="checkbox" id="showpoly" checked> polygons (p)</label>
  <span id="npoly" style="color:#fff"></span>
  <button id="fit">fit (f)</button>
  <span id="loading">loading&#8230;</span>
  <span style="margin-left:auto;color:#888">&#8592;/&#8594; time &nbsp; n/s frame &nbsp; shift+drag = box zoom</span>
</div>
<div id="map"></div>
<script>
const map = L.map('map', {crs: L.CRS.Simple, minZoom: 6, maxZoom: 19,
                          zoomSnap: 0.25, zoomDelta: 0.5});
let seqs = null, frame = 'North', idx = 0;
let overlay = null, polyLayer = null, firstLoad = true;

function key() { return seqs[frame][idx]; }

function fmtDate(d) { return d.slice(0,4)+'-'+d.slice(4,6)+'-'+d.slice(6,8); }

async function load(fit) {
  const k = key();
  document.getElementById('loading').style.display = 'inline';
  const meta = await (await fetch('/api/meta/' + k)).json();
  const b = [[meta.extent[0], meta.extent[1]], [meta.extent[2], meta.extent[3]]];
  const cmap = document.getElementById('cmap').value;
  const url = '/api/image/' + k + '?cmap=' + cmap;

  const img = new window.Image();
  img.onload = () => {
    if (overlay) map.removeLayer(overlay);
    overlay = L.imageOverlay(url, b, {className:'pix'}).addTo(overlay ? map : map);
    if (fit || firstLoad) { map.fitBounds(b); firstLoad = false; }
    document.getElementById('loading').style.display = 'none';
  };
  img.src = url;

  const [s, e] = k.split('_');
  document.getElementById('title').textContent =
      fmtDate(s) + ' \\u2192 ' + fmtDate(e) + '  (' + frame + ')';
  document.getElementById('pos').textContent =
      (idx+1) + '/' + seqs[frame].length;

  if (polyLayer) { map.removeLayer(polyLayer); polyLayer = null; }
  const gj = await (await fetch('/api/polygs/' + k)).json();
  polyLayer = L.geoJSON(gj, {style: {color:'#ffffff', weight:2,
                                     fillColor:'#ffffff', fillOpacity:0.25}});
  const n = gj.features ? gj.features.length : 0;
  document.getElementById('npoly').textContent = n ? n + ' polygs' : 'no polygs';
  if (document.getElementById('showpoly').checked) polyLayer.addTo(map);
}

function step(d) {
  idx = Math.min(Math.max(idx + d, 0), seqs[frame].length - 1);
  load(false);
}

function switchFrame(f) {
  if (f === frame || !seqs[f].length) return;
  const cur = key().slice(0, 8);
  frame = f;
  // jump to the nearest start date in the other frame
  let best = 0, bestDiff = Infinity;
  seqs[frame].forEach((k, i) => {
    const diff = Math.abs(parseInt(k.slice(0,8)) - parseInt(cur));
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  });
  idx = best;
  document.querySelector('input[name=frame][value='+f+']').checked = true;
  load(true);
}

document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('fit').onclick = () => load(true);
document.getElementById('cmap').onchange = () => load(false);
document.getElementById('showpoly').onchange = (e) => {
  if (!polyLayer) return;
  e.target.checked ? polyLayer.addTo(map) : map.removeLayer(polyLayer);
};
document.querySelectorAll('input[name=frame]').forEach(r =>
  r.onchange = () => switchFrame(r.value));
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') step(-1);
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'n' || e.key === 'N') switchFrame('North');
  else if (e.key === 's' || e.key === 'S') switchFrame('South');
  else if (e.key === 'f' || e.key === 'F') load(true);
  else if (e.key === 'p' || e.key === 'P') {
    const c = document.getElementById('showpoly');
    c.checked = !c.checked; c.dispatchEvent(new Event('change'));
  }
});

fetch('/api/intfs').then(r => r.json()).then(d => {
  seqs = d;
  if (!seqs[frame].length) frame = 'South';
  load(true);
});
</script>
</body>
</html>
"""


# ----------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            parts = url.path.strip('/').split('/')
            if url.path == '/':
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif url.path == '/api/intfs':
                self._send(200, json.dumps(SEQS).encode(), 'application/json')
            elif parts[:2] == ['api', 'meta'] and len(parts) == 3:
                body = json.dumps({'extent': intf_extent(parts[2])}).encode()
                self._send(200, body, 'application/json')
            elif parts[:2] == ['api', 'polygs'] and len(parts) == 3:
                self._send(200, polygons_geojson(parts[2]).encode(), 'application/json')
            elif parts[:2] == ['api', 'image'] and len(parts) == 3:
                cmap = parse_qs(url.query).get('cmap', ['gray'])[0]
                path = render_image(parts[2], cmap)
                prerender_neighbors(parts[2], cmap)
                with open(path, 'rb') as f:
                    data = f.read()
                ctype = 'image/jpeg' if ARGS.fmt == 'jpg' else 'image/png'
                self._send(200, data, ctype)
            else:
                self._send(404, b'not found', 'text/plain')
        except BrokenPipeError:
            pass
        except Exception as e:
            logging.exception('error serving %s', self.path)
            self._send(500, str(e).encode(), 'text/plain')

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; we log renders ourselves


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', ARGS.port), Handler)
    logging.info('viewer running: http://localhost:%d', ARGS.port)
    logging.info('from your laptop:  ssh -L %d:localhost:%d <cluster-host>',
                 ARGS.port, ARGS.port)
    server.serve_forever()
