to run the model on a new interferogram:
run predic_new_intf.py with command line args:

--intfs_dir', default='./', help='interferograms dir')
--intfs_list: A list of intf names in format YYYYmmdd_YYYYmmdd seperated by comma'
--model_dir': path to model directory
--model_file': the .pth model file in the model dir
--patch_size: a tuple of default=(200,100): patch H, patch W. ***These have to match the patch size in training that built the model ***. if it is not the default it will say so in the model's name.
--strdpp: default = 2, strides per patch in both directions
--plot_polygs: default=False. option to view the predicted polygons on top of the input interferogram. set True only if you have graphics on your machine
--output_polygs_dir': directory where the polyg files will be saved
--rth: reconstruction threshold, default=0.25
--add_gt_polygs: default=False, if there are gt polygons for the intf you can view them in the plot
--gt_polygons_file_path: default='sub_20231001.shp', path to the gt polygs file


files needed:
unet.py
unet_parts.py
the .pth model file (to locate in the --model_dir) (will be sent separately)
intf_coords.json
the gt polygon file if --add_gt_polygs is True.
lidar_mask_polygs.shp


