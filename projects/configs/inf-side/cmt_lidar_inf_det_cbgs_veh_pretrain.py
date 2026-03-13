# Infrastructure detection with vehicle-compatible spatial config for loading vehicle pretrain.
# Root cause: Inf mAP=0.01 vs Veh mAP=0.49 mainly due to load_from=None (train from scratch).
# Inf sparse_shape [41,576,640] differs from Veh [41,576,576], so cannot load Veh ckpt directly.
# This config aligns spatial config to vehicle so we can load cmt_lidar_det_epoch48.pth.

_base_ = ['./cmt_lidar_inf_det_cbgs.py']

# Override: use vehicle spatial config (enables load_from)
point_cloud_range = [0.0, -46.08, -3.0, 92.16, 46.08, 1.0]
post_center_range = [0.0, -56.08, -5.0, 102.16, 56.08, 5.0]
voxel_size = [0.16, 0.16, 0.1]
grid_size = [576, 576, 40]
sparse_shape = [41, 576, 576]

# Crop inf data to x<=92m, z<=1m; eval also in this range
class_range = {"car": 50, "pedestrian": 50, "bicycle": 50}
new_range_100 = False

# Load vehicle checkpoint (same sparse_shape/grid as vehicle)
load_from = 'ckps/cmt_lidar_det_epoch48.pth'
# load_from = None  # uncomment to train from scratch for comparison
