#!/usr/bin/env python3
"""
test_full_intf_v2.py — Predict & reconstruct interferograms.

Coordinate fix vs test_full_intf.py:
  - reconstruct_intf_prediction() now takes offset_x (no more hardcoded 3000).
  - Reads patch_coords.json produced by prepare_intrfrgrm_pathches_v2.py to get
    the exact geographic origin (x0_patch, y0_patch) of each intf's patch array.
  - All lon/lat-sensitive operations (LiDAR rasterization, polygon conversion, plot
    extents) use that origin directly with offset_x=0 — no second offset application.
"""

import os, sys, json, pickle, logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

import torch

from unet import UNet
from attn_unet import AttentionUNet
from polygs import plg_indx2longlat, mask_array_to_polygons
from get_intf_info import get_intf_coords, find_11day_sequences

def str2bool(arg: str) -> bool:
    return str(arg).strip().lower() == "true"

import numpy as np
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import rasterize


def reconstruct_intf_prediction(
    data,                   # (C, ny, nx, H, W) or (ny, nx, H, W) ; channel 0 = current
    intf_coords,            # tuple from get_intf_coords / your tuple logic
    net,                    # torch model
    patch_size,             # (patch_H, patch_W)
    stride,                 # int
    rth,                    # float threshold for reconstructed_pred_th
    mask=None,              # (ny, nx, H, W) or None (already cropped to data)
    add_lidar_mask=True,
    plot=False,
    lidar_sources=None,     # list[str|None] of length C; if None -> replicate current for all
    overlay_on_preds=True,
    device=None,
    treat_nodata_regions=False,
    blend=None,             # None | 'hann'
    pred_blend_dtype='float32',
    window_gamma=1.0,
    blend_space='prob',
    # ------------------------------------------------------------------
    # offset_x: number of columns by which the patch array is shifted
    # relative to x0 in intf_coords.  Pass 0 when intf_coords already
    # carries x0_patch (i.e. x0_frame + offset_x*dx computed in prepare).
    # ------------------------------------------------------------------
    offset_x=0,
):
    """
    Returns (if mask is not None):
      reconstructed_intf_all, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs
    Else:
      reconstructed_intf, reconstructed_pred, reconstructed_prevs
    """
    x0, y0, dx, dy, ncells, nlines, x4000, x8500, current_lidar_mask, num_nonz_p, bo, frame = intf_coords
    patch_H, patch_W = patch_size

    # Geographic longitude of reconstruction pixel column 0.
    x_recon_origin = x0 + offset_x * dx

    if data.ndim == 5:
        C, ny, nx, H, W = data.shape
        data_stack = data.astype(np.float32, copy=False)
    else:
        ny, nx, H, W = data.shape
        C = 1
        data_stack = data[None].astype(np.float32, copy=False)

    out_h = ny * patch_H // stride + patch_H * (1 - 1 // stride)
    out_w = nx * patch_W // stride + patch_W * (1 - 1 // stride)

    reconstructed_intf_all = np.zeros((C, out_h, out_w), dtype=np.float32)
    reconstructed_pred_raw = np.zeros((out_h, out_w), dtype=np.float32)
    reconstructed_mask     = np.zeros((out_h, out_w), dtype=np.float32) if mask is not None else None

    use_hann = (blend == 'hann')
    if use_hann:
        def get_hann_window(h, w, gamma=1.0):
            wy = np.hanning(h) if h > 1 else np.ones(1, dtype=np.float32)
            wx = np.hanning(w) if w > 1 else np.ones(1, dtype=np.float32)
            W  = (wy[:, None] * wx[None, :]).astype(np.float32)
            if gamma != 1.0:
                W = np.power(W, gamma, dtype=np.float32)
            return W / (W.max() + 1e-8)

        Wn = get_hann_window(patch_H, patch_W, gamma=window_gamma)
        dtype_np = np.float16 if str(pred_blend_dtype).lower() == 'float16' else np.float32
        Wn = Wn.astype(dtype_np, copy=False)
        pred_num = np.zeros((out_h, out_w), dtype=dtype_np)
        pred_den = np.zeros((out_h, out_w), dtype=dtype_np)

    rasterized_polygon_current = None
    mask_polyg_current = None
    mask_all = None

    if add_lidar_mask:
        lidar_gdf = gpd.read_file('lidar_mask_polygs.shp')
        if lidar_sources is None:
            srcs = [current_lidar_mask] * C
        else:
            srcs = list(lidar_sources)
            if len(srcs) != C:
                print(f"[warn] lidar_sources len {len(srcs)} != C {C}; using current for all")
                srcs = [current_lidar_mask] * C

        mask_all = np.zeros((C, out_h, out_w), dtype=np.uint8)
        # Use x_recon_origin (= x0 + offset_x*dx) as the geographic west edge of the
        # reconstructed image so the LiDAR mask is rasterized onto the correct grid.
        tr = rasterio.transform.from_origin(x_recon_origin, y0, dx, dy)

        for c, src in enumerate(srcs):
            if src is None or str(src).strip().lower() in ("", "none", "null"):
                polyg = lidar_gdf
            else:
                col = lidar_gdf['source'].astype(str).str.strip().str.lower()
                polyg = lidar_gdf[col == str(src).strip().lower()]
                if polyg.empty:
                    print(f"[warn] LiDAR source '{src}' not in shapefile -> using ALL polygons")
                    polyg = lidar_gdf

            mask_all[c] = rasterize(
                [(g, 1) for g in polyg['geometry'].tolist()],
                out_shape=(out_h, out_w),
                transform=tr,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
            if c == 0:
                rasterized_polygon_current = mask_all[c].copy()
                mask_polyg_current = polyg

    step_y = patch_H // stride
    step_x = patch_W // stride

    for i in range(ny):
        print(i)
        for j in range(nx):
            y0o = i * step_y; y1o = y0o + patch_H
            x0o = j * step_x; x1o = x0o + patch_W

            for c in range(C):
                reconstructed_intf_all[c, y0o:y1o, x0o:x1o] += data_stack[c, i, j] / (stride**2)

            if reconstructed_mask is not None:
                reconstructed_mask[y0o:y1o, x0o:x1o] += mask[i, j] / (stride**2)

            is_within_mask = True
            if add_lidar_mask and (mask_all is not None):
                is_within_mask = mask_all[:, y0o:y1o, x0o:x1o].all()

            if not is_within_mask:
                continue

            x_np = data_stack[:, i, j]

            if treat_nodata_regions:
                tol = 1e-9
                if np.isnan(x_np).any():
                    x_np = np.nan_to_num(x_np, nan=0.0)
                v_np = (np.abs(x_np - 0.5) > tol).astype(np.float32)
                x_np = np.concatenate([x_np, v_np], axis=0)

            image = torch.from_numpy(x_np[None]).to(device=device, memory_format=torch.channels_last)
            with torch.no_grad():
                logits = net(image)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

            if use_hann:
                pred_num[y0o:y1o, x0o:x1o] += (prob.astype(pred_blend_dtype) if pred_blend_dtype=='float16' else prob) * Wn
                pred_den[y0o:y1o, x0o:x1o] += Wn
            else:
                reconstructed_pred_raw[y0o:y1o, x0o:x1o] += prob / (stride**2)

    if use_hann:
        eps = 1e-8
        reconstructed_pred = (pred_num / (pred_den + eps)).astype(np.float32)
    else:
        reconstructed_pred = reconstructed_pred_raw

    reconstructed_pred_th = (reconstructed_pred > rth).astype(np.float32)
    reconstructed_intf    = reconstructed_intf_all[0]
    reconstructed_prevs   = reconstructed_intf_all[1:] if C > 1 else None

    if plot:
        extent = [x_recon_origin,
                  x_recon_origin + dx * out_w,
                  y0 - dy * out_h,
                  y0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.02)

        def overlay_current(ax):
            if not add_lidar_mask or rasterized_polygon_current is None:
                return
            ax.imshow(rasterized_polygon_current, extent=extent, cmap='Reds',
                      vmin=0, vmax=1, alpha=0.30, interpolation='nearest', zorder=2)
            if mask_polyg_current is not None:
                for geom in mask_polyg_current.geometry:
                    if geom is None: continue
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom)
                    for poly in polys:
                        xg, yg = poly.exterior.xy
                        ax.plot(xg, yg, color='yellow', linewidth=1.2, zorder=3)

        ax1.imshow(reconstructed_intf, extent=extent, cmap='gray', zorder=1)
        overlay_current(ax1)
        ax1.set_title('Reconstructed (current) + LiDAR')

        ax2.imshow(reconstructed_pred_th, extent=extent, cmap='binary', vmin=0, vmax=1, zorder=1)
        if overlay_on_preds:
            overlay_current(ax2)
        ax2.set_title('Prediction + LiDAR')

        for ax in (ax1, ax2):
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout(); plt.show()

    if reconstructed_mask is None:
        return reconstructed_intf, reconstructed_pred, reconstructed_prevs
    else:
        return reconstructed_intf_all, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs

# ----------------------------- args ------------------------------
def get_pred_args():
    import argparse
    p = argparse.ArgumentParser(description='Predict & reconstruct interferograms')
    p.add_argument('--full_intf_dir', type=str, default='./')
    p.add_argument('--input_patch_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/')
    p.add_argument('--plot_data', type=bool, default=False)
    p.add_argument('--patch_size', nargs='+', type=int, default=[200, 100])
    p.add_argument('--days_diff', type=int, default=11)

    p.add_argument('--intf_source', type=str, default='intf_list', choices=['intf_list', 'test_dataset', 'preset', 'all'])
    p.add_argument('--intf_list', type=str, default=None)
    p.add_argument('--test_dataset', type=str, default=None)
    p.add_argument('--valset_from_partition', type=str, default=None)

    p.add_argument('--model', '-m', default='checkpoint_epoch28.pth')
    p.add_argument('--data_stride', type=int, default=2)
    p.add_argument('--recon_th', type=float, default=0.25)
    p.add_argument('--job_name', type=str, default='job')
    p.add_argument('--add_lidar_mask', type=str, default='True')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--attn_unet', action='store_true')
    p.add_argument('--years_22_23', action='store_true')
    p.add_argument('--unioned_mask', action='store_true')
    p.add_argument('--k_prevs', type=int, default=0)
    p.add_argument('--treat_nodata_regions', action='store_true')
    p.add_argument('--blend_type', type=str, default=None, choices=['hann', 'center-crop'])
    p.add_argument('--window_gamma', type=float, default=1.0)

    return p.parse_args()

# ----------------------------- main ------------------------------
if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'
    logging.basicConfig(level=logging.INFO)

    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)
    now = datetime.now().strftime("%m_%d_%Hh%M")
    args = get_pred_args()
    args.add_lidar_mask = str2bool(args.add_lidar_mask)
    if not is_running_locally:
        args.plot = False

    job_name   = f"{args.job_name}_{now}"
    model_name = args.model.split('.')[0]
    output_path      = f'pred_outputs2/{model_name}/{job_name}/'
    output_polyg_dir = os.path.join(output_path, 'polygs/')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_polyg_dir, exist_ok=True)

    log_file = os.path.join(output_path, f"{args.job_name}_{now}.log")
    logging.getLogger().addHandler(logging.FileHandler(log_file))

    logging.info(f'Script: {os.path.abspath(__file__)}')
    logging.info('Running job {} with model {}.pth and args: {}'.format(job_name, model_name, args))

    if (args.intf_source == 'intf_list' and args.intf_list is None) or \
       (args.intf_source == 'test_dataset' and args.test_dataset is None) or \
       (args.intf_source == 'preset' and args.valset_from_partition is None):
        logging.info('you chose {} but it is None. exiting.'.format(args.intf_source))
        sys.exit(1)

    if args.intf_source == 'intf_list':
        intf_list = args.intf_list.split(',')
    elif args.intf_source == 'preset':
        with open(args.valset_from_partition, 'r') as f:
            loaded = json.load(f)
        intf_list = loaded['val']
    elif args.intf_source == 'all':
        intf_list = []
    else:
        with open('./test_data/' + args.test_dataset, 'rb') as f:
            test_data = pickle.load(f)
        intf_list = test_data.ids
    num_c = args.k_prevs + 1

    if args.treat_nodata_regions:
        num_c = num_c * 2

    net = UNet(n_channels=num_c, n_classes=1, bilinear=False)
    if args.attn_unet:
        net = AttentionUNet(n_channels=num_c, n_classes=1, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model} on device {device}')
    net.to(device=device)
    state_dict = torch.load('./models/' + args.model, map_location=device)
    _ = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict); net.eval()

    patch_H, patch_W = args.patch_size
    data_dir = os.path.join(
        args.input_patch_dir,
        f'data_patches_H{patch_H}_W{patch_W}_strpp{args.data_stride}'
        f'_{args.days_diff}days' + ('_22_23' if args.years_22_23 else '') + '_Aligned'
    )
    mask_dir = os.path.join(
        args.input_patch_dir,
        f'mask_patches_H{patch_H}_W{patch_W}_strpp{args.data_stride}'
        f'_{args.days_diff}days' + ('_22_23' if args.years_22_23 else '') + '_Aligned'
    )

    # Load the coordinate metadata saved by prepare_intrfrgrm_pathches_v2.py.
    # It records x0_patch and y0_patch: the exact geographic lon/lat of pixel (col=0, row=0)
    # in the patch array for each interferogram.
    coords_json_path = os.path.join(data_dir, "patch_coords.json")
    if not os.path.exists(coords_json_path):
        logging.error(
            f"patch_coords.json not found in {data_dir}.\n"
            "Run prepare_intrfrgrm_pathches_v2.py to generate it."
        )
        sys.exit(1)
    with open(coords_json_path, 'r') as f:
        patch_coords_dict = json.load(f)

    if args.intf_source == 'all':
        intf_list = sorted(set(
            f[13:30] for f in os.listdir(data_dir) if f.endswith('.npy')
        ))

    prev_dict = None
    if args.k_prevs > 0:
        with open('intf_coord.json', "r") as f:
            intf_info = json.load(f)
        prev_dict, updated = find_11day_sequences(intf_info, k_prev=args.k_prevs, restrict_to=intf_list)
        intf_list = updated
        if args.years_22_23:
            intf_list = [
                k for k in intf_list
                if all(p.startswith('2022') for p in prev_dict[k]['prevs'])
            ]

    tol = 1e-3

    for intf in intf_list:
        ic = get_intf_coords(intf)

        if intf not in patch_coords_dict:
            logging.warning(f"{intf} not in patch_coords.json — skipping.")
            continue

        pinfo = patch_coords_dict[intf]
        # x0_patch, y0_patch: geographic lon/lat of reconstruction array column 0, row 0.
        # These already incorporate the frame alignment and the column offset applied
        # during patchification (prepare_intrfrgrm_pathches_v2.py).
        x0_patch  = pinfo["x0_patch"]
        y0_patch  = pinfo["y0_patch"]
        dx        = pinfo["dx"]
        dy        = pinfo["dy"]

        # Build intfs_coords with x0_patch as x0 so that reconstruct_intf_prediction
        # uses the correct geographic origin with offset_x=0.
        intfs_coords = (x0_patch, y0_patch) + ic[2:]

        data_fn = f'data_patches_{intf}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy'
        mask_fn = f'mask_patches_{intf}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy'
        data_path = os.path.join(data_dir, data_fn)
        mask_path = os.path.join(mask_dir, mask_fn)

        cur = np.load(data_path).astype(np.float32)

        if args.k_prevs > 0:
            prev_ids = prev_dict[intf]['prevs'][:args.k_prevs][::-1]
            prevs = [
                np.load(os.path.join(data_dir, f'data_patches_{pid}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy')
                       ).astype(np.float32)
                for pid in prev_ids
            ]
            pa = [cur] + prevs

            ny = min(p.shape[0] for p in pa)
            nx = min(p.shape[1] for p in pa)
            Hc, Wc = pa[0].shape[2], pa[0].shape[3]

            normed = []
            for p in pa:
                pc = p[:ny, :nx, :Hc, :Wc].astype(np.float32, copy=False)
                mn, mx = float(np.nanmin(pc)), float(np.nanmax(pc))
                if (mn >= -tol) and (mx <= 1.0 + tol):
                    q = pc.copy()
                    zmask = (q == 0.0)
                    if zmask.any(): q[zmask] = 0.5
                    normed.append(q)
                else:
                    normed.append((pc + np.pi) / (2 * np.pi))
            data = np.stack(normed, axis=0).astype(np.float32)

        else:
            mn, mx = float(np.nanmin(cur)), float(np.nanmax(cur))
            if (mn >= -tol) and (mx <= 1.0 + tol):
                q = cur.copy()
                zmask = (q == 0.0)
                if zmask.any(): q[zmask] = 0.5
                data = q[None]
            else:
                data = ((cur + np.pi) / (2 * np.pi))[None]
            ny, nx, Hc, Wc = cur.shape

        mask_cur = np.load(mask_path).astype(np.float32)
        if args.k_prevs > 0:
            prev_mask_paths = [
                os.path.join(mask_dir, f'mask_patches_{pid}_H{patch_H}_W{patch_W}_strpp{args.data_stride}.npy')
                for pid in prev_ids
            ]
            prev_masks = [np.load(p).astype(np.float32) for p in prev_mask_paths]
            ma = [mask_cur] + prev_masks
            ma = [m[:ny, :nx, :Hc, :Wc] for m in ma]
            mask = (np.stack(ma, axis=0) > 0).any(axis=0).astype(np.float32) if args.unioned_mask \
                   else ma[0]
        else:
            mask = mask_cur

        logging.info(f"{mask_fn} shape used: {mask.shape}")

        cur_lm = ic[8]
        lidar_sources = [cur_lm]
        if args.k_prevs > 0:
            for pid in prev_ids:
                pic = get_intf_coords(pid)
                lidar_sources.append(pic[8])

        # offset_x=0 because intfs_coords already has x0=x0_patch (the patch origin).
        out = reconstruct_intf_prediction(
            data, intfs_coords, net, (patch_H, patch_W),
            args.data_stride, args.recon_th, mask,
            add_lidar_mask=args.add_lidar_mask,
            plot=False,
            lidar_sources=lidar_sources,
            overlay_on_preds=True,
            device=device,
            treat_nodata_regions=args.treat_nodata_regions,
            blend=args.blend_type,
            window_gamma=args.window_gamma,
            offset_x=0,
        )

        if mask is None:
            reconstructed_intf, reconstructed_pred, reconstructed_prevs = out
            reconstructed_mask = None
            reconstructed_pred_th = (reconstructed_pred > args.recon_th).astype(np.float32)
        else:
            reconstructed_intf_all, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs = out
            reconstructed_intf = reconstructed_intf_all[0]

        # x0_patch is the geographic longitude of reconstruction column 0.
        # Polygon pixel indices map directly to lon/lat via x0_patch + col*dx, y0_patch - row*dy.
        polygons = plg_indx2longlat(
            mask_array_to_polygons(reconstructed_pred_th),
            intfs_coords,
            x_start=x0_patch,
        )
        shp_path = os.path.join(output_polyg_dir, f'{intf}_predicted_polygs.shp')
        polygons.to_file(shp_path)

        prefix = os.path.join(output_path, intf)
        if mask is None:
            if reconstructed_prevs is None:
                image_to_save = reconstructed_intf[None].astype(np.float32)
            else:
                image_to_save = np.concatenate(
                    [reconstructed_intf[None], reconstructed_prevs], axis=0
                ).astype(np.float32)
        else:
            image_to_save = reconstructed_intf_all.astype(np.float32)
        if args.intf_source != 'all':
            np.save(prefix + '_image', image_to_save)
            np.save(prefix + '_pred_th', reconstructed_pred_th)
            np.save(prefix + '_pred', reconstructed_pred)
            if reconstructed_mask is not None:
                np.save(prefix + '_gt', reconstructed_mask)

        if args.plot:
            k_prev = 0 if reconstructed_prevs is None else reconstructed_prevs.shape[0]
            has_mask = reconstructed_mask is not None
            ncols = 1 + k_prev + (1 if has_mask else 0) + 1
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5), sharex=True, sharey=True)
            axes = np.atleast_1d(axes)

            extent = [x0_patch,
                      x0_patch + dx * reconstructed_intf.shape[1],
                      y0_patch - dy * reconstructed_intf.shape[0],
                      y0_patch]

            col = 0
            axes[col].imshow(reconstructed_intf, extent=extent, cmap='jet'); axes[col].set_title('Current'); col += 1
            if k_prev > 0:
                for p in range(k_prev):
                    axes[col].imshow(reconstructed_prevs[p], extent=extent, cmap='jet')
                    axes[col].set_title(f'Prev t-{p+1}'); col += 1
            if has_mask:
                axes[col].imshow(reconstructed_mask, extent=extent); axes[col].set_title('True mask'); col += 1
            axes[col].imshow(reconstructed_pred_th, extent=extent, cmap='binary', vmin=0, vmax=1); axes[col].set_title('Pred mask')

            plt.tight_layout(); plt.show()
