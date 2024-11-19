#!/usr/bin/env python3

# A code for processing interferograms and identifiying subsidence due to sinkholes
# By: Gali Dekel @ Weizmann Institute, 2024
# Contrib: Ran Novitsky Nof @ Geological Survey of Israel, 2024

import argparse
import os
import sys
from unet import *
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.warp
import rasterio.features
import fiona
from skimage.util import view_as_windows
from logging.handlers import TimedRotatingFileHandler
from numba import njit, prange
from alive_progress import alive_bar

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')

# Default Parameters
VERBOSE = False
LOG_FILE = None
LOG_LEVEL = 'INFO'
PTHMODEL = './models/subsidence.pth'  # model path and file
PATCHSIZE = [200, 100]  # Patch size (H, W), do not change - match to model trianing data
BATCHSIZE = 1  # number of patches to process at the same time
STRIDESTEP = 2  # Patch stride step. width (or height) devided by the stride step value (in pixels)
RTH = 0.25  # percentage of predictions per pixel threshold.
NUMCORES = int(os.cpu_count() * 0.75) or 1 # multi processing
log = logging.getLogger('SubsidenceML')
log.setLevel(LOG_LEVEL)  # set at default log level


def restricted_float(x):
    """Help function for argparse to get a 0-1 float"""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} is not a valid float")
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x

# Command line arguments:
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Predict subsidence polygons for new interferogram',
    epilog='''Created by Gali Dekel (gali.dekel@weizmann.ac.il), 2024 @ Wizmann''')
# process verbosity
parser.add_argument('-v', help=f'verbose - print messages to screen? (Default: {VERBOSE})', action='store_true', default=VERBOSE)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL,
                    help=f"Log level (Default: {LOG_LEVEL}). see Python's Logging module for more details")
parser.add_argument('--logfile', metavar='log file name', help='log to file (Default: no file)', default=LOG_FILE)
# Inputs
parser.add_argument('interferograms', metavar='Interfrogram', type=str, nargs = '*', help=f'interferogram for processing')
parser.add_argument('-z', '--mask', metavar='maskfile', type=str, help=f'A mask raster. Any rasterio valid format. NoValue or 0 will be used as mask. Optional', default=None)
parser.add_argument('-m', '--model', metavar='model.pth', type=str, help=f'the .pth model file. Default: {PTHMODEL}', default=PTHMODEL)
# Outputs
parser.add_argument('-o', '--output', metavar='Polygons_dir_path', type=str, help='Polygons shape files directory', default=None)
# Processing
parser.add_argument('-s', '--strdpp', metavar='step', type = int, default = STRIDESTEP, help=f'Strides per patch in both directions. Default: {STRIDESTEP}')
parser.add_argument('-p', '--patch_size', nargs=2, type=int, metavar=('H', 'W'), default=PATCHSIZE, help=f'Height and Width of a patch (in pixels). Must match the training patch size. Default: {PATCHSIZE}')
parser.add_argument('-b', '--batch_size', type=int, metavar='size', default=BATCHSIZE, help=f'Processing batch size. Use for GPU. Default: {BATCHSIZE}')
parser.add_argument('--rth', metavar='', type=restricted_float, default=RTH, help=f'Stride composite threshold value (0-1). Default {RTH}')
parser.add_argument('-c', '--cores', metavar='NCORES', type=int, default=NUMCORES, help=f"Number of CPU corse for parallel processing. Default (75%%) of availbale cores: {NUMCORES}")
# Plotting
parser.add_argument('--plot',  action='store_true', default=False)
# Testing
parser.add_argument('--memtest', action='store_true', help='Test batch size memory limits for GPU and exit. Use --batch_szie for a starting size', default=False)


def set_multithreads(ncores):
    # make sure we have at least one core and the value is integer
    ncores = int(ncores) or 1
    # Set the number of intra-op threads to the number of CPU cores
    torch.set_num_threads(ncores)
    # Set the number of inter-op threads (typically 1 to prevent oversubscription)
    torch.set_num_interop_threads(1)
    # Set environment variables for underlying libraries
    os.environ["OMP_NUM_THREADS"] = f"{ncores}"
    os.environ["MKL_NUM_THREADS"] = f"{ncores}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{ncores}"
    # Set GDAL configuration options
    os.environ['GDAL_NUM_THREADS'] = f"4"
    os.environ['CPL_CPU_COUNT'] = f"4"
    os.environ['VSI_CACHE'] = 'TRUE'


def get_batch_size(model, input_shape, device="cuda", start_batch_size=1, step=1):
    """
    Automatically determines the maximum batch size that fits in memory.
    
    Args:
        model (torch.nn.Module): The model to test.
        input_shape (tuple): Shape of a single input sample (C, H, W).
        device (str): Device to test on ("cuda" or "cpu").
        start_batch_size (int): Starting batch size for testing.
        step (int): Increment to try larger batch sizes.
    
    Returns:
        int: Maximum batch size that fits in memory.
    """

    # For CPU, better to use only one patch at a time
    if device != 'cuda':
        log.warnning(f'For CPU please use the default batch size (1)')
        return 1
    
    batch_size = start_batch_size
    model = model.to(device)
    log.info(f'Performing memory test for batch zise')
    while True:
        try:
            # Create a dummy input batch
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Perform a forward pass
            with torch.no_grad():
                model(dummy_input)
            
            # If successful, increment batch size
            log.debug(f'Passed: {batch_size}')
            batch_size += step
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Out of memory error, return the last successful batch size
                log.info(f'Maximal available batch size: {batch_size - step} on device')
                torch.cuda.empty_cache()  # Clear the cache if using GPU
                return batch_size - step
            else:
                # Other errors
                raise e


def from_raster(filename, readdata=False):
    with rasterio.open(filename) as dataset:
        dx, _, x0, _, dy, y0, *_ = list(dataset.transform)
        nrows, ncols = dataset.shape
        if readdata:
            log.debug('Reading dataset')
            data = dataset.read(1)
        else:
            data = None
    try:
        master, slave = os.path.split(filename)[1].split('.')[0].split('_')[2:4]
    except:
        master, slave = 'm', 's'
    return master, slave, nrows, ncols, x0, y0, dx, dy, data


def to_raster(filename, data, dx, x0, dy, y0, CRS=4326):
    """
    Creates an ERS raster file from a 2D NumPy array.

    Args:
        filename (str): The output file name without extension.
        data (numpy.ndarray): 2D array of raster data (MxN).
        dx (float): Pixel width (resolution in x-direction).
        x0 (float): X-coordinate of the upper-left corner.
        dy (float): Pixel height (resolution in y-direction, usually negative).
        y0 (float): Y-coordinate of the upper-left corner.
    
    Returns:
        rasterio.io.DatasetReader: The rasterio dataset object of the created raster.
    """
    # Ensure the data is a 2D NumPy array
    if data.ndim != 2:
        raise ValueError("Data must be a 2D NumPy array.")
    # Determine the height and width of the raster
    height, width = data.shape
    # Create an affine transform
    transform = rasterio.transform.from_origin(x0, y0, dx, dy)
    # Define the coordinate reference system (CRS)
    # Modify the CRS according to your data's projection
    crs = rasterio.CRS.from_epsg(CRS)  # WGS84 Latitude/Longitude

    # Set the metadata for the raster
    metadata = {
        'driver': 'ERS',
        'height': height,
        'width': width,
        'count': 1,  # Number of bands
        'dtype': data.dtype,
        'crs': crs,
        'transform': transform
    }

    # Output file path with .ers extension
    output_file = f"{filename}.ers"

    # Write the data to the ERS file
    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(data, 1)  # Write data to the first band

    # Open the file again to return the dataset object
    filename = output_file
    return filename


def get_mask(interferogram, maskfile):
    """
    Creates a numpy array of ones based on the dimensions of the base raster
    and updates it with zeros where the mask (raster or shapefile) indicates.
    
    Parameters:
        interferogram (str): Path to the base raster file.
        maskfile (str): Path to the mask raster or shapefile.
    
    Returns:
        numpy.ndarray: Array of ones and zeros with dimensions of the interferogram.
    """
    # Open the interferogram and read its properties
    with rasterio.open(interferogram) as raster:
        dim = (raster.height, raster.width)
        transform = raster.transform
        crs = raster.crs
        if crs is None:
            crs = rasterio.CRS.from_epsg(4326)
        dtype = np.uint8  # Using uint8 for binary mask
    
    # Create an array of ones with the dimensions of the base raster
    masked = np.ones(dim, dtype=np.uint8)
    
    try:
        # The mask is a raster file
        with rasterio.open(maskfile) as mask:
            # Read the mask data
            mask_data = mask.read(1)
            mask_transform = mask.transform
            mask_crs = mask.crs
            if mask_crs is None:
                mask_crs = rasterio.CRS.from_epsg(4326)
            mask_nodata = mask.nodata
            
        # Prepare an array to receive the reprojected mask data
        reprojected_mask = np.empty(dim, dtype=mask_data.dtype)
        
        # Reproject the mask raster to match the base raster's grid
        rasterio.warp.reproject(
            source=mask_data,
            destination=reprojected_mask,
            src_transform=mask_transform,
            src_crs=mask_crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=rasterio.warp.Resampling.bilinear,
            dst_nodata=mask_nodata
        )
        
        # Create a boolean mask where mask data is zero or NoData
        if mask_nodata is not None:
            invalid_mask = (reprojected_mask == 0) | (reprojected_mask == mask_nodata)
        else:
            invalid_mask = (reprojected_mask == 0)
    except rasterio.RasterioIOError:    
        # The mask is a shapefile or other vector format
        # Open the shapefile
        with fiona.open(maskfile, 'r') as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            mask_crs = shapefile.crs
            # Handle the case where the shapefile's CRS is missing
            if mask_crs is None:
                mask_crs = rasterio.CRS.from_epsg(4326)
        # Check if the shapefile's CRS matches the base raster's CRS
        if mask_crs != crs:
            raise ValueError("The shapefile's CRS does not match the base raster's CRS.")
        # Rasterize the shapefile to match the base raster's grid
        reprojected_mask = rasterio.features.rasterize(
            shapes,
            out_shape=dim,
            transform=transform,
            fill=0,
            out=None,
            all_touched=False,
            dtype=dtype
        )
        # Create a boolean mask where rasterized data is zero
        invalid_mask = (reprojected_mask == 0)
    # Update the output array: set zeros where the invalid_mask is True
    masked[invalid_mask] = 0
    return masked


def patchify(interferogram, patch_size, stride, maskfile= None):
    """Convert data into patches of patch size, with overlaps of strid
    Input:
        interferogram: Raster file path. data will be (MxN 2D numpy.array)
        patch_size: H, W of patch
        stride: dy, dx of steps
        data_mask: MxN 2D numpy.array of 1 and 0 mask or None if not available
    Output:
        masked_patches: ncol x nrows 4D numpy.array with masked input_array data, devided into HxW patches
        nonz: ncol x nrows 2D numpy.array with maps of True or False for data in patches.

        nrows = (M - H) // dy + 1
        ncols = (N - W) // dx + 1

    """
    # Read The data and metadata
    master, slave, nrows, ncols, x0, y0, dx, dy, data = from_raster(interferogram, True)
    log.debug(f'Interferogram size: {nrows}x{ncols} rowsxcols')
    if maskfile is not None:
        log.debug('Creating mask')
        data_mask = get_mask(interferogram, maskfile)
        assert data_mask.shape == data.shape, "Mask array should be the same shape as input array"
        data_mask *= np.where(data, 1, 0).astype(np.uint8)
    else:
        data_mask = np.where(data, 1, 0)
    # Mask & Normlize
    data = ((data + np.pi) / (2 * np.pi)) * data_mask
    to_raster(f'tgeo_mask_{master}_{slave}.unw', data, dx, x0, dy, y0)
    # devide to patches
    patches = view_as_windows(data, patch_size, stride)
    log.debug(f'Mapped {patches.shape[0]}x{patches.shape[1]} patches of {patches.shape[2]}x{patches.shape[3]} pixels')
    nonz = patches.any(axis=(2,3))
    log.debug(f'Found {nonz.sum()} valid patches out of {np.prod(nonz.shape)}')
    return patches, nonz


def process_patches(data, mask, net, device="cpu", batch_size=16):
    """
    Process MxN patches of HxW through a U-Net model on GPU or CPU.

    Args:
        data (numpy.ndarray): Input data as an MxN array of HxW patches.
        mask (numpy.ndarray): MxN boolean mask, where True indicates a patch with data.
        net (torch.nn.Module): Trained U-Net model.
        device (str): Device to use ("cpu" or "cuda").
        batch_size (int): Number of patches to process in one batch.
        

    Returns:
        numpy.ndarray: Predictions with the same shape as the input data.
    """

    # Flatten the first two dimensions for easier iteration over patches
    flat_data = data.reshape(-1, data.shape[2], data.shape[3])
    flat_mask = mask.flatten()
    # Initialize predictions array
    predictions = np.zeros_like(flat_data, dtype=np.float32)
    valididx = np.where(flat_mask)[0]
    validpred = np.zeros_like(flat_data)

    # Batch processing
    baroff = logging.StreamHandler not in [h.__class__ for h in log.handlers]
    with alive_bar(len(valididx), spinner='classic', disable=baroff, dual_line=True) as bar:
        bar.text = f'Processing {len(valididx) // batch_size} batches of {batch_size} patches on {device}'
        for idx in range(0, len(valididx), batch_size):
            # Collect patches for the current batch
            batch_patches = flat_data[idx:idx + batch_size]
            
            # Convert batch to a PyTorch tensor and move to the appropriate device
            memf = torch.channels_last if device == "cuda" else torch.preserve_format
            batch_tensor = torch.tensor(batch_patches).unsqueeze(1).to(
                device=device, dtype=torch.float32, memory_format=memf
            )            
            # Forward pass through the model
            preds = net(batch_tensor)            
            # Apply sigmoid activation and thresholding
            preds = (torch.nn.functional.sigmoid(preds) > 0.5).float()            
            # Remove batch and channel dimensions and move back to CPU for numpy conversion
            preds = preds.squeeze(1).cpu().detach().numpy()
            # Store predictions in the corresponding indices
            validpred[idx:idx + batch_size] = preds
            bar()
    
    # Map the valid predictions back to the patches
    predictions[flat_mask] = validpred[flat_mask]
    predictions = predictions.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    return predictions


@njit(parallel=True)
def reconstruct_from_patches_numba(nrows, ncols, patches, stride, rth):
    """
    Reconstruct MxN patches of HxW into a cumulative value.

    Args:
        nrows (int): Number of rows in the interferogram data.
        ncols (int): Number of columns in the interferogram data.
        patches (numpy.ndarray): Input data as an MxN array of HxW patches.
        stride (int): Steps in pixels of the patch window.
        rth (float): Threshold value for classification.
        
    Returns:
        numpy.ndarray: Predictions with the same shape as the interferogram data.
    """    
    H, W = patches.shape[2], patches.shape[3]
    reconstructed = np.zeros((nrows, ncols), dtype=patches.dtype)
    
    ys = np.arange(0, nrows - H + 1, stride[0])
    xs = np.arange(0, ncols - W + 1, stride[1])
    
    # Use prange for parallel loops
    for i in prange(len(ys)):
        y = ys[i]
        for j in prange(len(xs)):
            x = xs[j]
            reconstructed[y:y + H, x:x + W] += patches[i, j] / 4  # 4 is args.strdpp**2
    
    # Thresholding
    for i in prange(reconstructed.shape[0]):
        for j in prange(reconstructed.shape[1]):
            if reconstructed[i, j] > rth:
                reconstructed[i, j] = 1.0
            else:
                reconstructed[i, j] = 0.0

    return reconstructed.astype(np.float32)    


def reconstruct_from_patches(nrows, ncols, patches, stride, rth):
    """
    Reconstruct MxN patches of HxW into a cumulative value.

    Args:
        interferogram (str): Inteferogrma file name
        patches (numpy.ndarray): Input data as an MxN array of HxW patches.
        stride: Steps in pixel of the patch window
        rth: sigmoid threshold for predicted subsidence
        
    Returns:
        numpy.ndarray: Predictions with the same shape as the interferogram data.
    """    

    H, W = patches.shape[2], patches.shape[3]
    reconstructed = np.zeros([nrows, ncols], dtype=patches.dtype)
    
    ys = np.arange(0, nrows - H + 1, stride[0])
    xs = np.arange(0, ncols - W + 1, stride[1])
    
    for i in range(len(ys)):
        for j in range(len(xs)):
            y = ys[i]
            x = xs[j]
            reconstructed[y:y + H, x:x + W] += patches[i, j] / 4  # 4 is args.strdpp**2
    
    reconstructed = np.where(reconstructed > rth, 1, 0)
    return reconstructed.astype(np.float32)


def reconstruct(output, interferogram, patches, stride, rth, use_numba=True):
    master, slave, nrows, ncols, x0, y0, dx, dy, *_ = from_raster(interferogram, False)
    filename = output + os.sep + 'tgeo_pred_' + master + '_' + slave + '.unw'
    if use_numba:
        log.debug('Using Numba')
        reconstructed = reconstruct_from_patches_numba(nrows, ncols, patches, stride, rth)
    else:
        log.debug('Using NumPy')
        reconstructed = reconstruct_from_patches(nrows, ncols, patches, stride, rth)
    log.debug(f'Saving reconstructed to file: {filename}')
    to_raster(filename, reconstructed, dx, x0, dy, y0)
    return filename


def get_polygons(predictedfile):
    # Open the raster file
    with rasterio.open(predictedfile + '.ers') as src:
        raster = src.read(1)
        mask = raster == 1  # Create a mask for values equal to 1
        transform = src.transform
        crs = src.crs
        if crs is None:
            crs = rasterio.CRS.from_epsg(4326)
    # Extract shapes where mask is True
    shapes_generator = rasterio.features.shapes(raster, mask=mask, transform=transform)
    # Prepare the schema
    schema = {
        'geometry': 'Polygon',
        'properties': {'value': 'int'},
    }
    # Write to shapefile
    log.debug(f'saving shapefile of predicted polygons to {predictedfile}.shp')
    with fiona.open(predictedfile + '.shp', 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
        for geom, val in shapes_generator:
            shp.write({
                'geometry': geom,
                'properties': {'value': int(val)},
            })
    return predictedfile + '.shp'


def set_logger(log, verbose=VERBOSE, log_level=LOG_LEVEL, logfile=LOG_FILE):
    log.setLevel(log_level)
    if verbose:
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        if logging.StreamHandler not in [h.__class__ for h in log.handlers]:
            log.addHandler(ch)
        else:
            log.warning('log Stream handler already applied.')
    if logfile:
        # create file handler
        fh = TimedRotatingFileHandler(logfile,
                                      when='midnight',
                                      utc=True)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if TimedRotatingFileHandler not in [h.__class__ for h in log.handlers]:
            log.addHandler(fh)
        else:
            log.warning('Log file handler already applied.')
        log.info(f'Log file is: {logfile}')
    else:
        log.debug(f'No Log file was set')


if __name__ == '__main__':
    plt.switch_backend('Qt5Agg')
    args = parser.parse_args()
    if not args.memtest and not args.interferograms:
        parser.error("the following arguments are required: interferograms (unless --memtest is specified)")
    set_logger(log, args.v, args.log_level, args.logfile)
    #log.debug(args)
    # GPU? CPU?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.debug(f'Using {device}')
    if device=='cpu' and args.batch_size > 1:
        log.warning(f'For CPU batch size should be 1. Please consider to re-run without the --batch_size parameter')
    # Multi-threading
    set_multithreads(args.cores)
    log.debug(f"Intra-op threads: {torch.get_num_threads()}")
    log.debug(f"Inter-op threads: {torch.get_num_interop_threads()}")
    if args.output is None:
        args.output = '.'
    output = os.path.abspath(args.output)
    if not os.path.exists(output):
        log.warning(f'{output} Does not exists. attempting to create it')
        try:
            os.makedirs(output)
        except Exception as EX:
            log.critical(f'Output folder creation faild: {EX}\nTerminating.')
            sys.exit(f'Output folder creation faild: {EX}\nTerminating.')
    log.debug(f'Saving polygons to: {output}')
    log.debug(f'Using Patches {args.patch_size[0]}X{args.patch_size[1]}, with strides of {args.strdpp}')
    stride = (args.patch_size[0]//args.strdpp, args.patch_size[1]//args.strdpp)
    log.debug(f'Moving patches window by {stride[0]}X{stride[1]} pixels')
    # create the torch.nn.Module
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.to(device=device)
    log.debug(f'Loading model parameters from {args.model}')
    state_dict = torch.load(args.model, weights_only=True, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])  # remove records of trainning
    try:
        net.load_state_dict(state_dict)
    except Exception as EX:
        log.critical(f'Mismatch model patameters: {EX}')
        sys.exit()
    net.eval()
    log.info('Model loaded!')
    # Test memory to get the maximal batch size and exit
    if args.memtest:
        H, W = args.patch_size
        bs = get_batch_size(net, (1, H, W), device=device, start_batch_size=500, step=10)
        print(bs)
        sys.exit(0)
    # or process interferograms
    log.info(f'Processing {len(args.interferograms)} new interferogram{'s' if len(args.interferograms)>1 else ''}')
    for i, interferogram in enumerate(args.interferograms):
        log.info(f'Processing: {interferogram}')
        log.debug('Creating patches')    
        patches, nonz = patchify(interferogram, args.patch_size, stride, maskfile=args.mask)
        log.debug('Get Predictions')
        predictions = process_patches(patches, nonz, net, device, args.batch_size)
        log.debug('Reconstructing predictions from patches. Using score > {args.rth}')
        predictionfile = reconstruct(args.output, interferogram, predictions, stride, args.rth)
        log.debug('Extracting prediction polygons')
        polygonsfile = get_polygons(predictionfile)