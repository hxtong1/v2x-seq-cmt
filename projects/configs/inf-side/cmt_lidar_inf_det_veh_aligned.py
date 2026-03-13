"""Infra detection with vehicle-aligned pc_range & sparse_shape for loading vehicle ckp.
   Same spatial config as vehicle so ckp can load; uses infra data (infrastructure-side).
   Purpose: verify if vehicle pretrain can improve infra mAP.
"""
_base_ = './cmt_lidar_inf_det_cbgs.py'

# Override with vehicle spatial config for checkpoint compatibility
point_cloud_range = [0.0, -46.08, -3.0, 92.16, 46.08, 1.0]
post_center_range = [0.0, -56.08, -5.0, 102.16, 56.08, 5.0]
voxel_size = [0.16, 0.16, 0.1]
grid_size = [576, 576, 40]
sparse_shape = [41, 576, 576]

# Eval filter distance aligned with vehicle range
class_range = dict(car=50, pedestrian=50, bicycle=50)
new_range_100 = False

# Load vehicle checkpoint (must match sparse_shape)
load_from = '/home/thx/data-mnt/code/CMT/ckps/cmt_lidar_det_epoch48.pth'
