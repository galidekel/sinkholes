
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
    num_nonz_p = intf_coords[intf_name]['nonz_num']
    bo = intf_coords[intf_name]['byte_order']
    frame = intf_coords[intf_name]['frame']

    x4000 = x0 + 4000*dx
    x8500 = x4000 + 4500*dx

    return (x0, y0, dx, dy,ncells, nlines, x4000, x8500,lidar_mask,num_nonz_p,bo,frame)
def get_intf_lidar_mask(intf_name):
    with open('lidar_intf_mask.txt', 'r') as f:
        mask = 'no_mask'
        for line in f:
            if intf_name[:8] == line[8:16] and intf_name[9:17] == line[24:32]:
                mask = line[40:49]
            elif intf_name[:8] == line[8:16]:
                mask = line[40:49]


    return mask

import numpy as np
def crop_to_start_xy(intf: np.ndarray,
                     mask: np.ndarray,
                     x0: float, y0: float,          # source top-left (easting, northing)
                     x_star: float, y_star: float,  # desired top-left to crop to
                     dx: float = 2.777e-05,
                     dy: float = 2.777e-05,
                     tol: float = 1e-2):
    """
    Crop intf and mask so that the output starts at (x_star, y_star).
    Optionally crop to a specific (width, height). If width/height are None,
    returns everything from that start to the image edge.

    Coordinates follow rasterio.from_origin(x0, y0, dx, dy):
      - x increases to the RIGHT
      - y increases DOWNWARD in array indices; georeferenced 'northing' decreases by dy per row.

    Args:
      intf, mask: 2D arrays with identical shape (H, W)
      x0, y0:   source top-left easting/northing
      x_star, y_star: desired new top-left easting/northing
      dx, dy:   pixel size in degrees (or meters), positive
      width, height: optional size of the crop in pixels
      tol:      tolerance for sub-pixel rounding (in coordinate units)

    Returns:
      intf_c, mask_c, new_x0, new_y0, (row_off, col_off)
    """
    if intf.shape != mask.shape:
        raise ValueError(f"intf and mask must have same shape, got {intf.shape} vs {mask.shape}")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    H, W = intf.shape

    # Compute integer offsets that align x_star, y_star to pixel grid
    col_f = (x_star - x0) / dx
    row_f = (y0 - y_star) / dy  # note y decreases as row increases

    col_off = int(round(col_f))
    row_off = int(round(row_f))

    # Check sub-pixel mismatch
    if abs(col_f - col_off) * dx > tol or abs(row_f - row_off) * dy > tol:
        raise ValueError(
            f"(x*,y*) not on pixel grid within tolerance: "
            f"col_off ~ {col_f} (dx={dx}), row_off ~ {row_f} (dy={dy}). "
            f"Consider snapping x*/y* to grid."
        )

    # Bounds check for start
    if not (0 <= col_off < W) or not (0 <= row_off < H):
        raise ValueError(f"Requested start is outside the image: row_off={row_off}, col_off={col_off}, shape={intf.shape}")

    # Determine crop size
    if width is None:
        width = W - col_off
    if height is None:
        height = H - row_off

    # Bounds check for end
    if col_off + width > W or row_off + height > H:
        raise ValueError(
            f"Requested crop (row_off={row_off}, col_off={col_off}, height={height}, width={width}) "
            f"exceeds image bounds {intf.shape}"
        )

    # Do the crop
    intf_c = intf[row_off:row_off + height, col_off:col_off + width]
    mask_c = mask[row_off:row_off + height, col_off:col_off + width]

    # New georeferenced top-left equals the requested (x*, y*)
    new_x0 = x0 + col_off * dx  # should equal x_star within tol
    new_y0 = y0 - row_off * dy  # should equal y_star within tol

    return intf_c, mask_c, new_x0, new_y0, (row_off, col_off)
import math
from typing import Dict, Any, Iterable, Tuple

def _extent_from_meta(m: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return (left, right, top, bottom) from one meta dict."""
    x0 = float(m["east"])
    y0 = float(m["north"])
    dx = float(m["dx"])
    dy = float(m["dy"])
    W  = int(m["ncells"])
    H  = int(m["nlines"])
    left   = x0
    right  = x0 + dx * W
    top    = y0
    bottom = y0 - dy * H
    return left, right, top, bottom

def _check_consistent_spacing(metas: Iterable[Dict[str, Any]], tol=1e-12) -> Tuple[float, float]:
    """Verify dx, dy consistent across metas; return (dx,dy)."""
    dxs = {round(float(m["dx"]), 15) for m in metas}
    dys = {round(float(m["dy"]), 15) for m in metas}
    if len(dxs) != 1 or len(dys) != 1:
        raise ValueError(f"Inconsistent spacing: dxs={dxs}, dys={dys}. Resample first.")
    dx = float(list(dxs)[0]); dy = float(list(dys)[0])
    if dx <= 0 or dy <= 0: raise ValueError("Non-positive dx/dy")
    return dx, dy

def _intersection_bbox(extents: Iterable[Tuple[float,float,float,float]]):
    """Intersect a list of (left,right,top,bottom)."""
    extents = list(extents)
    if not extents:
        raise ValueError("No extents to intersect")
    left   = max(e[0] for e in extents)
    right  = min(e[1] for e in extents)
    top    = min(e[2] for e in extents)   # 'top' is the largest latitude; intersection uses min of tops
    bottom = max(e[3] for e in extents)
    if right <= left or top <= bottom:
        raise ValueError("No overlapping area across frames in this region.")
    return left, right, top, bottom

def _is_on_grid(x_star: float, x0: float, step: float, atol_pix=1e-9) -> bool:
    """Check (x_star - x0)/step is (near) integer."""
    col = (x_star - x0) / step
    return abs(col - round(col)) <= atol_pix

def build_common_grid_for_region(
    meta_dict: Dict[str, Dict[str, Any]],
    frame_label: str,                         # "North" or "South"
    keys: Iterable[str] | None = None,        # optional subset of keys
    tol_pix: float = 0.01
) -> Tuple[float, float, int, int, float, float]:
    """
    From meta JSON, compute a common crop grid for a region:
      returns (x_star, y_star, width, height, dx, dy)

    - Uses the INTERSECTION of all frames' extents in that region.
    - Ensures (x_star, y_star) lies on the pixel grid of *every* frame.
    """
    # pick region frames
    if keys is None:
        keys = [k for k, m in meta_dict.items() if m.get("frame") == frame_label]
    else:
        keys = [k for k in keys if meta_dict[k].get("frame") == frame_label]
    if not keys:
        raise ValueError(f"No frames with frame='{frame_label}'")

    metas = [meta_dict[k] for k in keys]

    # check spacing consistency
    dx, dy = _check_consistent_spacing(metas)

    # intersect extents
    extents = [_extent_from_meta(m) for m in metas]
    left_i, right_i, top_i, bottom_i = _intersection_bbox(extents)

    # choose target top-left as intersection's left & top
    x_star = left_i
    y_star = top_i

    # verify x_star/y_star align to each frame's grid
    for m in metas:
        x0 = float(m["east"])
        y0 = float(m["north"])
        if not _is_on_grid(x_star, x0, dx, atol_pix=tol_pix) or not _is_on_grid(y_star, y0, dy, atol_pix=tol_pix):
            raise ValueError(
                "Common origin not aligned to all frames' pixel grids.\n"
                f"  offending frame: x0={x0}, y0={y0}\n"
                f"  candidate x*={x_star}, y*={y_star}, dx={dx}, dy={dy}\n"
                "If this persists, you have mixed grids; resample or round x0/y0 to grid."
            )

    # compute pixel width/height from intersection
    width  = int(math.floor((right_i  - x_star) / dx + 1e-9))
    height = int(math.floor((y_star   - bottom_i) / dy + 1e-9))
    if width <= 0 or height <= 0:
        raise ValueError("Degenerate common grid (zero width/height).")

    # snap right/bottom back to grid exactly (informative; not strictly needed for cropping)
    right_snapped  = x_star + width  * dx
    bottom_snapped = y_star - height * dy
    # Optional: you can log differences (right_i - right_snapped), etc.

    return x_star, y_star, width, height, dx, dy
import json

intf_dict_file = open('intf_coord.json', 'r')
intf_coords = json.load(intf_dict_file)
new_grid = build_common_grid_for_region(intf_coords,'North')
print(new_grid)

if __name__ == '__main__':
    intf = '20190730_20190810'
    mask = get_intf_lidar_mask(intf)
    print(mask)
###