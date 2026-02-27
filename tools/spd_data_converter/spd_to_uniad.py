import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from concurrent.futures import ThreadPoolExecutor
from math import pi
import os.path as osp
import argparse
import json
import random
import string
from tqdm import tqdm
import uuid
from scipy.linalg import polar
from tools.spd_data_converter.spd_prediction_tools import get_forecasting_annotations
import os, psutil
from queue import Queue
QUEUE_MAXSIZE = 64
# GlOBAL 

TOTAL_CPU = os.cpu_count()
RESERVE_CPU = 2  # 系统保留
MAX_AVAILABLE = TOTAL_CPU - RESERVE_CPU

NUM_PROCESSES = min(MAX_AVAILABLE, 8)  # 每帧进程数
WORKERS_PER_PROCESS = max(1, MAX_AVAILABLE // NUM_PROCESSES)

to_remove_list_veh = ['015389', '015390', '015391', '015392', '015393', '015394', '015395', '015396', '015397', '015398', '015399', '004394', '004395', '004396', '004397', '004398', '004399', 
             '004400', '004401', '004402', '004403', '004404', '004405', '004406', '004407', '004408', '004409', '004411', '004412', '004413', '002060', '002061', '002062', '002063', 
             '002064', '013006', '013083', '013084', '013085', '013086', '013087', '013088', '013089', '013090', '013091', '013092', '013093', '013094', '013155', '013156', '013157', 
             '013158', '013159', '013160', '013161', '013162', '013163', '013164', '013165', '013166', '013167', '013168', '013169', '013170', '013171', '013172', '013173', '013175',
             '013186', '013187', '013188', '013189', '013206', '013207', '013208', '013209', '012263', '012264', '012265', '012266', '012268', '012269', '012270', '012271', '012272', 
             '012273', '012274', '012275', '012276', '012277', '012278', '012279', '012280', '012281', '012282', '012283', '012284', '012285', '012286', '012287', '012288', '012289', '012290', 
             '003211', '003212', '003213',
             '003214', '003215', '003216', '003217', '003218', '003219', '003220', '003221', '003222', '003223', '003224', '003225', '003226', '003227', '003228',
             '003229', '003230', '004262', 
             '004263', '004264', '004265', '004266', '004381', '004382', '004383', '004386', '004388', '004389', '011118', '011119', '011120', '011121', '011122', '011123', '011124', 
             '011125', '011126', '011127', '011128', '011129', '011130', '011131', '011132', '011133', '011134', '011135', '011136', '011137', '011138', '011139', '011140', '011141', 
             '011142', '011143', '011144', '011145', '011146', '011147', '011148', '011149', '011150', '005596', '005597', '005598', '005599', '005600', '005601', '005602', '005603', 
             '005604', '005605', '005606', '005607', '005608', '005609', '005610', '004845', '004846', '004847', '004848', '004849', '004850', '004851', '004852', '006600', '006602', 
             '006603', '006604', '006605', '006606', '006607', '006608', '006609', '006610', '006611', '006612', '006613', '006614', '006615', '006616', '006617', '006618', '005538', 
             '005539', '005540', '005541', '005542', '005543', '005544', '005545', '014466', '014467', '014468', '008626', '008627', '008628', '008629', '008630', '008631', '008632', 
             '008633', '008634', '008757', '008758', '008759', '008760', '014551', '014552', '014553', '014554', '014555', '014556', '014570', '014571', '014572', '014573', '014574', 
             '014575', '014576', '014577', '006378', '006379', '006380', '006381', '006382', '006383', '006384', '006385', '006386', '006387', '006388', '006389', '006390', '006391', 
             '006392', '006393', '006394', '006395', '006396', '000368', '000369', '000370', '000372', '000373', '000375', '000377', '000378', '000379', '000380', '000381', '000382', 
             '000383', '000384', '000385', '000386', '000387', '000388', '000390', '000391', '000392', '000393', '000394', '000395', '000396', '000400', '000401', '003593', '003594', '003595', '003596', 
             '003597', '003598', '003599', '003600', '003601', '003602', '003603', '003604', '007457', '007458', '007459', '007460', '007461', '007462', '007463', '007464', '007465', 
             '007466', '007467', '007468', '007469', '007470', '007471', '007472', '007473', '007474', '007475', '007476', '007477', '007478', '007479', '007480', '007481', '007482', '009103', 
             '009104', '009105', '009106', '009107']

to_remove_list_coop = ['003211', '003212', '003213', '003214', '003215', '003216', '003217', '003218', '003219', '003220', '003221', '003222', '003223', '003224', '003225', '003226', '003227', '003228', '003229', '003230', 
 '004394', '004395', '004396', '004397', '004398', '004399', '004400', '004401', '004402', '004403', '004404', '004405', '004406', '004407', '004408', '004409', '004410', '004411', '004412', '004413', 
 '007457', '007458', '007459', '007460', '007461', '007462', '007463', '007464', '007465', '007466', '007467', '007468', '007469', '007470', '007471', '007472', '007473', '007474', '007475', '007476', '007477', '007478', '007479', '007480', '007481', '007482', 
 '013006', '013083', '013084', '013085', '013086', '013087', '013088', '013089', '013090', '013091', '013092', '013093', '013094', '013206', '013207', '013208', '013209']

# --- Core Utility Functions ---

def orthonormalize_rotation(R):
    """ 
    Fix non-orthogonal rotation matrix issues using SVD. 
    (Replaces the mathematically incorrect ICP call for 3x3 matrices).
    """
    U, S, Vt = np.linalg.svd(R)
    return U @ Vt

def get_transformation_matrix(rotation_quat, translation_vec):
    """ 
    Synthesize a 4x4 homogeneous transformation matrix from a quaternion and translation.
    """
    T = np.eye(4)
    # Convert Quaternion to 3x3 Rotation Matrix
    T[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    T[:3, 3] = np.array(translation_vec).flatten()
    return T

def transform_points(points, T):
    """ 
    Coordinate transformation for point clouds: P_target = T * P_source.
    Optimized via vectorized matrix multiplication.
    """
    # Create homogeneous coordinates: (N, 3) -> (N, 4)
    points_homo = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    # Apply transformation using the transpose for efficiency: (N, 4) @ (4, 4).T
    return (points_homo @ T.T)[:, :3]

def get_cam_intr(calib_path):
    try:
        intr = np.array(load_json(calib_path)['P']).reshape(3, 4)[:, :3]
    except:
        intr = np.array(load_json(calib_path)['cam_K']).reshape(3, 3)

    return intr


def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    """
        R_new = R2 @ R1
        t_new = R2 @ t1 + t2
    """
    # 
    r1 = np.asarray(rotation_1)
    t1 = np.asarray(translation_1).flatten()  # 强制转为一维 (3,)
    r2 = np.asarray(rotation_2)
    t2 = np.asarray(translation_2).flatten()  # 强制转为一维 (3,)

    # 使用 @ 运算符进行矩阵乘法
    # R2 @ R1: 旋转相乘
    rotation = r2 @ r1
    
    # R2 @ t1 + t2: 平移向量变换
    translation = (r2 @ t1) + t2

    return rotation, translation


# ---- Core Config Parameters
class Box3D():
    def __init__(self):
        self.center = None
        self.wlh = None
        self.orientation_yaw_pitch_roll = None
        self.name = None
        self.token = None
        self.instance_token = None
        self.track_id = None
        self.prev_token = None
        self.next_token = None
        self.timestamp = None
        self.visibility = None
        self.gt_velocity = None
        self.prev = None
        self.next = None

visibility_mappings = {
    0: 4,
    1: 3,
    2: 2,
    3: 1
}

## UniV2X TODO: remapping
## UniV2X TODO: modify the related code in UniAD and nuScenes
class_names_nuscenes_mappings = {
    'Car': 'car',
    'Truck': 'car',
    'Van': 'car',
    'Bus': 'car',
    'Motorcyclist': 'bicycle',
    'Cyclist': 'bicycle',
    'Tricyclist': 'bicycle',
    'Barrowlist': 'bicycle',
    'Pedestrian': 'pedestrian',
    'TrafficCone': 'traffic_cone',
    'car': 'car',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone'
}


#---------Core base DataProcess Function-----

def generate_json_maps_files(data_root, version='v1.0-mini'):
    json_types = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                  'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

    import shutil
    if not os.path.exists(osp.join(data_root, version)):
        tmp_nuscenes_json_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/v1.0-mini'
        shutil.copytree(tmp_nuscenes_json_root, osp.join(data_root, version))

    if not os.path.exists(osp.join(data_root, 'maps')):
        tmp_nuscenes_map_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/maps'
        shutil.copytree(tmp_nuscenes_map_root, osp.join(data_root, 'maps'))

def gen_token(*args):
    token_name = ''
    for value in args:
        token_name += str(value)
    token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
    return str(token)


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data


def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)


# ------------Core transfomer matrix Functions

def get_lidar_ego_global_infos(root_path, data_infos, v2x_side):
    """
    获取每帧的 LiDAR -> Ego 和 Ego -> Global 信息
    使用 SVD 正交化 rotation，替代原先 ICP
    """

    lidar_ego_global_infos = {}

    for data_info in tqdm(data_infos):
        sample_token = data_info['frame_id']
        lidar_ego_global_infos[sample_token] = {}

        if v2x_side == 'infrastructure-side':
            # lidar -> ego 默认 identity
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.eye(3, dtype=np.float32)
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.zeros(3, dtype=np.float32)

            # ego -> global
            calib_path = osp.join(root_path, data_info['calib_virtuallidar_to_world_path'])
            calib = load_json(calib_path)

            R = np.array(calib['rotation'], dtype=np.float32)
            t = np.array(calib['translation'], dtype=np.float32).reshape(3)

            # 使用 SVD 正交化
            R = orthonormalize_rotation(R)
            q = Quaternion(matrix=R)

            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = t

        else:  # vehicle-side
            # LiDAR -> Ego
            calib_l2e_path = osp.join(root_path, data_info['calib_lidar_to_novatel_path'])
            calib_l2e = load_json(calib_l2e_path)
            R_l2e = orthonormalize_rotation(np.array(calib_l2e['transform']['rotation'], dtype=np.float32))
            t_l2e = np.array(calib_l2e['transform']['translation'], dtype=np.float32).reshape(3)
            q_l2e = Quaternion(matrix=R_l2e)
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array([q_l2e.w, q_l2e.x, q_l2e.y, q_l2e.z], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = t_l2e

            # Ego -> Global
            calib_e2g_path = osp.join(root_path, data_info['calib_novatel_to_world_path'])
            calib_e2g = load_json(calib_e2g_path)
            R_e2g = orthonormalize_rotation(np.array(calib_e2g['rotation'], dtype=np.float32))
            t_e2g = np.array(calib_e2g['translation'], dtype=np.float32).reshape(3)
            q_e2g = Quaternion(matrix=R_e2g)
            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array([q_e2g.w, q_e2g.x, q_e2g.y, q_e2g.z], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = t_e2g

        # 4x4 transformation matrices
        lidar_ego_global_infos[sample_token]['T_lidar2ego'] = get_transformation_matrix(
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'],
            lidar_ego_global_infos[sample_token]['lidar2ego_translation']
        )
        lidar_ego_global_infos[sample_token]['T_ego2global'] = get_transformation_matrix(
            lidar_ego_global_infos[sample_token]['ego2global_rotation'],
            lidar_ego_global_infos[sample_token]['ego2global_translation']
        )

    return lidar_ego_global_infos

def cal_ego_velocity(data_infos, sample_info_mappings, lidar_ego_global_infos):
    ego_velocity = {}

    n = len(data_infos)

    for i, data_info in enumerate(data_infos):
        token = data_info['frame_id']

        cur_loc = np.asarray(
            lidar_ego_global_infos[token]['ego2global_translation'],
            dtype=np.float32
        )
        cur_time = float(sample_info_mappings[token]['timestamp']) / 1e6

        # ---------- 第一帧 ----------
        if i == 0:
            next_token = data_infos[i+1]['frame_id']
            next_loc = np.asarray(
                lidar_ego_global_infos[next_token]['ego2global_translation'],
                dtype=np.float32
            )
            next_time = float(sample_info_mappings[next_token]['timestamp']) / 1e6
            dt = next_time - cur_time
            ego_velocity[token] = (next_loc - cur_loc)[:2] / max(dt, 1e-6)

        # ---------- 最后一帧 ----------
        elif i == n - 1:
            prev_token = data_infos[i-1]['frame_id']
            prev_loc = np.asarray(
                lidar_ego_global_infos[prev_token]['ego2global_translation'],
                dtype=np.float32
            )
            prev_time = float(sample_info_mappings[prev_token]['timestamp']) / 1e6
            dt = cur_time - prev_time
            ego_velocity[token] = (cur_loc - prev_loc)[:2] / max(dt, 1e-6)

        # ---------- 中间帧 ----------
        else:
            prev_token = data_infos[i-1]['frame_id']
            next_token = data_infos[i+1]['frame_id']

            prev_loc = np.asarray(
                lidar_ego_global_infos[prev_token]['ego2global_translation'],
                dtype=np.float32
            )
            next_loc = np.asarray(
                lidar_ego_global_infos[next_token]['ego2global_translation'],
                dtype=np.float32
            )

            prev_time = float(sample_info_mappings[prev_token]['timestamp']) / 1e6
            next_time = float(sample_info_mappings[next_token]['timestamp']) / 1e6

            dt = next_time - prev_time
            ego_velocity[token] = (next_loc - prev_loc)[:2] / max(dt, 1e-6)

    return ego_velocity

def generate_sweeps(sample_token, sample_info_mappings, data_infos_mapping, max_sweeps=10):
    sweeps = []
    cur_timestamp = float(sample_info_mappings[sample_token]['timestamp'])
    prev_token = sample_info_mappings[sample_token]['prev']

    while len(sweeps) < max_sweeps and prev_token != '' and prev_token in data_infos_mapping:
        sweep_info = data_infos_mapping[prev_token]
        rel_time = (cur_timestamp - float(sweep_info['timestamp'])) / 1e6
        sweeps.append({
            'lidar_path': sweep_info['lidar_path'].replace('.pcd', '.bin'),
            'timestamp': rel_time,
            'lidar2ego_rotation': sweep_info['lidar2ego_rotation'],
            'lidar2ego_translation': sweep_info['lidar2ego_translation'],
        })
        prev_token = sample_info_mappings[prev_token]['prev']
    return sweeps

def process_can_bus(ego_translation, ego_rotation):
    can_bus = np.zeros(18)

    def quaternion_yaw(q: Quaternion):
        """Extract yaw angle (rotation around Z axis) from a quaternion."""
        # q: pyquaternion.Quaternion
        w, x, y, z = q.elements  # pyquaternion order: w, x, y, z
        # yaw formula: atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw
    can_bus[:3] = ego_translation
    can_bus[3:7] = ego_rotation  # quaternion
    yaw = quaternion_yaw(Quaternion(ego_rotation))  # radians
    can_bus[-2] = yaw
    can_bus[-1] = yaw  # 可保留最后一位，方便 debug
    return can_bus


# ---------- Interpolation Function-------------------
def loc_linear_interpolation(
                            loc_ii_0, 
                            loc_ii_1, 
                            timestamp_ii_0, 
                            timestamp_ii_1, 
                            cur_timestamp,
                            lidar_ego_global_info_0,
                            lidar_ego_global_info_1,
                            cur_lidar_ego_global_info):
    """Use linear interpolation to estimate the 3d location for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    #cvt to global
    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['lidar2ego_rotation']).rotation_matrix, center_0) \
                 + np.array(lidar_ego_global_info_0['lidar2ego_translation'])
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['ego2global_rotation']).rotation_matrix, center_0) \
                 + np.array(lidar_ego_global_info_0['ego2global_translation'])

    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])

    #cvt to global
    center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['lidar2ego_rotation']).rotation_matrix, center_1) \
                        + np.array(lidar_ego_global_info_1['lidar2ego_translation'])
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['ego2global_rotation']).rotation_matrix, center_1)\
                        + np.array(lidar_ego_global_info_1['ego2global_translation'])

    #global interpolation
    ratio = (cur_timestamp - timestamp_ii_0)/(timestamp_ii_1 - timestamp_ii_0)
    cur_center = (1 - ratio) * center_0 + ratio * center_1

    #cur sesor data interpolation
    global2ego_r = np.linalg.inv(Quaternion(cur_lidar_ego_global_info['ego2global_rotation']).rotation_matrix)
    global2ego_t = -np.array(cur_lidar_ego_global_info['ego2global_translation']).reshape(1,3) @ global2ego_r.T

    ego2lidar_r = np.linalg.inv(Quaternion(cur_lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix)
    ego2lidar_t = -np.array(cur_lidar_ego_global_info['lidar2ego_translation']).reshape(1,3) @ ego2lidar_r.T

    cur_center = np.dot(global2ego_r, cur_center) + global2ego_t.reshape(3)
    cur_center = np.dot(ego2lidar_r, cur_center) + ego2lidar_t.reshape(3)
    locs_out = {'x': cur_center[0], 'y': cur_center[1], 'z': cur_center[2]}
    
    return locs_out

def rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp):
    """Use linear interpolation to estimate the rotation for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    # cur_rot  = (rot_ii_1 - rot_ii_0) / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + rot_ii_0

    q0 = Quaternion(axis=[0,0,1], angle=rot_ii_0)
    q1 = Quaternion(axis=[0,0,1], angle=rot_ii_1)

    ratio = (cur_timestamp - timestamp_ii_0)/(timestamp_ii_1 - timestamp_ii_0)
    cur_rot = Quaternion.slerp(q0, q1, ratio).yaw_pitch_roll[0]
    return cur_rot

#-----------Veloctity Process--------------

def _generate_unvisible_annotations(
                                    v2x_side, 
                                    sample_info_mappings, 
                                    secene_frame_mappings,
                                    instance_token_mappings, 
                                    total_annotations, 
                                    lidar_ego_global_infos,
                                    max_workers=WORKERS_PER_PROCESS
                                    ):
    """
    对不可见目标进行插值补全位置（visibility 为 occluded 的对象）。
    主要用于预测或速度计算。
    """
    output_queue = Queue(maxsize=QUEUE_MAXSIZE)
    def process_instance(instance_token, output_queue):
        cur_instance_samples = instance_token_mappings[instance_token]

        for ii in range(len(cur_instance_samples) - 1):
            cur_frame_idx = cur_instance_samples[ii]['frame_idx'] + 1
            while cur_frame_idx != cur_instance_samples[ii + 1]['frame_idx']:
                # 插值所需参数
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']
                rot_ii_0 = cur_instance_samples[ii]['annotation']['rotation']
                rot_ii_1 = cur_instance_samples[ii + 1]['annotation']['rotation']
                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']

                cur_sample_token = secene_frame_mappings[
                    (cur_instance_samples[0]['scene_token'], cur_frame_idx)
                ]

                # 插值位置和旋转
                cur_loc = loc_linear_interpolation(
                    loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1,
                    sample_info_mappings[cur_sample_token]['timestamp'],
                    lidar_ego_global_infos[cur_instance_samples[ii]['sample_token']],
                    lidar_ego_global_infos[cur_instance_samples[ii+1]['sample_token']],
                    lidar_ego_global_infos[cur_sample_token]
                )

                cur_rot = rot_linear_interpolation(
                    rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1,
                    sample_info_mappings[cur_sample_token]['timestamp']
                )

                cur_anno_token = gen_token(
                    v2x_side, cur_sample_token,
                    str(cur_loc['x']), str(cur_loc['y']), str(cur_loc['z'])
                )

                cur_instance_sample_anno = {
                    "token": cur_anno_token,
                    "type": cur_instance_samples[ii]['annotation']['type'],
                    "track_id": cur_instance_samples[ii]['annotation']['track_id'],
                    "truncated_state": 0,
                    "occluded_state": 3,
                    "3d_dimensions": cur_instance_samples[ii]['annotation']['3d_dimensions'],
                    "3d_location": cur_loc,
                    "rotation": cur_rot,
                    "instance_token": instance_token
                }

                # 直接把结果放到队列里，不生成 dict
                output_queue.put((cur_sample_token, cur_anno_token, cur_instance_sample_anno))
                cur_frame_idx += 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_instance, inst, output_queue) for inst in instance_token_mappings]

        # 等待所有线程完成
        for f in as_completed(futures):
            f.result()  # 保证异常可以抛出

    # 主线程合并结果
    while not output_queue.empty():
        sample_token, anno_token, anno = output_queue.get()
        if sample_token not in total_annotations:
            total_annotations[sample_token] = {}
        total_annotations[sample_token][anno_token] = anno

    return total_annotations

def _add_annotation_velocity_prev_next(
                                        total_annotations, 
                                        instance_token_mappings, 
                                        lidar_ego_global_infos,
                                        max_workers=NUM_PROCESSES
                                        ):
    """
    Generate velocity and prev/next token for annotations in NuScenes style.

    Args:
        total_annotations (dict): {frame_id: {anno_token: anno_dict}}
        instance_token_mappings (dict): {instance_token: list of sample annotations}
        lidar_ego_global_infos (dict): {sample_token: lidar->ego, ego->global transforms}

    Returns:
        total_annotations, instance_token_mappings
    """
    def process_instance(instance_token, samples):
        updated_samples = []
        num_samples = len(samples)

        # 缓存旋转矩阵
        quat_cache = {}
        def get_global_center(sample_node):
            loc = sample_node['annotation']['3d_location']
            token = sample_node['sample_token']
            info = lidar_ego_global_infos[token]

            # cache quaternion
            if token not in quat_cache:
                quat_cache[token] = {
                    'l2e_r': Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                    'e2g_r': Quaternion(info['ego2global_rotation']).rotation_matrix,
                    'l2e_t': np.array(info['lidar2ego_translation']),
                    'e2g_t': np.array(info['ego2global_translation'])
                }

            q = quat_cache[token]
            center = np.dot(q['l2e_r'], [loc['x'], loc['y'], loc['z']]) + q['l2e_t']
            center = np.dot(q['e2g_r'], center) + q['e2g_t']
            return center

        for ii in range(num_samples):
            curr_sample = samples[ii]
            curr_token = curr_sample['sample_token']
            prev_anno_token = samples[ii - 1]['annotation']['token'] if ii > 0 else ''
            next_anno_token = samples[ii + 1]['annotation']['token'] if ii < num_samples - 1 else ''

            try:
                if ii < num_samples - 1:
                    t0 = float(curr_sample['timestamp']) / 1e6
                    t1 = float(samples[ii + 1]['timestamp']) / 1e6
                    p0 = get_global_center(curr_sample)
                    p1 = get_global_center(samples[ii + 1])
                elif num_samples > 1:
                    t0 = float(samples[ii - 1]['timestamp']) / 1e6
                    t1 = float(curr_sample['timestamp']) / 1e6
                    p0 = get_global_center(samples[ii - 1])
                    p1 = get_global_center(curr_sample)
                else:
                    t0, t1, p0, p1 = 0, 1, np.zeros(3), np.zeros(3)

                dt = t1 - t0
                gt_velocity = (p1 - p0) / dt if dt > 0 else np.zeros(3)

            except KeyError:
                gt_velocity = np.zeros(3)

            curr_sample['annotation']['gt_velocity'] = gt_velocity[:2].tolist()
            curr_sample['annotation']['prev'] = prev_anno_token
            curr_sample['annotation']['next'] = next_anno_token
            updated_samples.append(curr_sample)

        return (instance_token, updated_samples)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for instance_token, samples in instance_token_mappings.items():
            futures.append(executor.submit(process_instance, instance_token, samples))

        for fut in as_completed(futures):
            instance_token, updated_samples = fut.result()
            instance_token_mappings[instance_token] = updated_samples
            # 更新 total_annotations 对应每帧
            for sample_node in updated_samples:
                token = sample_node['sample_token']
                total_annotations[token][sample_node['annotation']['token']] = sample_node['annotation']

    return total_annotations, instance_token_mappings

# ----------Based batch sampler Process
def _get_secene_frame_mappings(sample_info_mappings):
    secene_frame_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        secene_frame_mappings[(scene_token, frame_idx)] = sample_token

    return secene_frame_mappings


def _get_instance_token_mappings(total_annotations, sample_info_mappings):
    instance_token_mappings = {}

    for sample_token, annotations in total_annotations.items():
        s_info = sample_info_mappings[sample_token]
        
        for anno_token, annotation in annotations.items():
            i_token = annotation["instance_token"]
            if i_token not in instance_token_mappings:
                instance_token_mappings[i_token] = []
            
            instance_token_mappings[i_token].append({
                'scene_token': s_info['scene_token'],
                'frame_idx': s_info['frame_idx'],
                'sample_token': sample_token,
                'timestamp': s_info['timestamp'],
                'annotation': annotation
            })

    # 修复点：使用 .sort() 进行就地排序，确保后续插值和速度计算顺序正确
    for i_token in instance_token_mappings:
        instance_token_mappings[i_token].sort(key=lambda x: x['frame_idx'])

    return instance_token_mappings

# ----------SPD datasets Process-------------------

def _generate_sample_infos(data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    sample_mappings = {d['frame_id']: d for d in data_infos}
    scene_dict = {}
    for d in data_infos:
        scene_token = d['sequence_id']
        scene_dict.setdefault(scene_token, []).append(d['frame_id'])

    sample_infos = []
    for scene_token, frame_ids in scene_dict.items():
        frame_ids.sort()
        for idx, fid in enumerate(frame_ids):
            info = {
                'token': fid,
                'timestamp': float(sample_mappings[fid]['pointcloud_timestamp']) / 1e6,  # 秒
                'image_timestamp': float(sample_mappings[fid]['image_timestamp']) / 1e6,
                'scene_token': scene_token,
                'location': sample_mappings[fid]['intersection_loc'],
                'frame_idx': idx,
                'prev': frame_ids[idx-1] if idx > 0 else '',
                'next': frame_ids[idx+1] if idx < len(frame_ids)-1 else ''
            }
            sample_infos.append(info)

    sample_info_mappings = {info['token']: info for info in sample_infos}

    return sample_infos, sample_info_mappings

def _generate_sample_infos_coop(coop_data_infos, veh_data_infos, inf_data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    veh_dict = {d['frame_id']: d for d in veh_data_infos}
    inf_dict = {d['frame_id']: d for d in inf_data_infos}
    
    scene_dict = {}
    coop_sample_mappings = {}
    
    for coop_data in coop_data_infos:
        veh_fid = coop_data['vehicle_frame']
        inf_fid = coop_data['infrastructure_frame']
        assert coop_data['vehicle_sequence'] == coop_data['infrastructure_sequence']
        coop_sample_mappings[veh_fid] = coop_data
        scene_dict.setdefault(coop_data['vehicle_sequence'], []).append(veh_fid)

    sample_infos = []
    for scene_token, frame_ids in scene_dict.items():
        frame_ids.sort()
        for idx, veh_fid in enumerate(frame_ids):
            inf_fid = coop_sample_mappings[veh_fid]['infrastructure_frame']
            veh_info = veh_dict[veh_fid]
            inf_info = inf_dict[inf_fid]
            coop_info = coop_sample_mappings[veh_fid]

            info = {
                'token': veh_fid,
                'timestamp': float(veh_info['pointcloud_timestamp']) / 1e6,
                'image_timestamp': float(veh_info['image_timestamp']) / 1e6,
                'scene_token': veh_info['sequence_id'],
                'location': veh_info['intersection_loc'],
                'frame_idx': idx,
                'prev': frame_ids[idx-1] if idx>0 else '',
                'next': frame_ids[idx+1] if idx<len(frame_ids)-1 else '',
                'token_inf': inf_fid,
                'timestamp_inf': float(inf_info['pointcloud_timestamp']) / 1e6,
                'image_timestamp_inf': float(inf_info['image_timestamp']) / 1e6,
                'system_error_offset': coop_info['system_error_offset']
            }
            sample_infos.append(info)
    
    sample_info_mappings = {info['token']: info for info in sample_infos}
    return sample_infos, sample_info_mappings

def _get_total_annotations(root_path, data_infos, sample_info_mappings, max_workers=NUM_PROCESSES):
    """
    Load all annotation JSONs for vehicle-side or infrastructure-side dataset.
    
    Args:
        root_path (str): 数据根路径
        data_infos (list): 已加载的 data_info 列表
        sample_info_mappings (dict): sample_token -> sample_info
        max_workers (int): 多线程读取 JSON 数量

    Returns:
        dict: {frame_id: {anno_token: anno_dict}}
    """

    def load_single_annotation(data_info):
        sample_token = data_info['frame_id']
        scene_token = sample_info_mappings[sample_token]['scene_token']
        annotation_path = osp.join(root_path, data_info['label_lidar_std_path'])
        annos = load_json(annotation_path)
        ann_dict = {}
        for anno in annos:
            track_id = anno.get('track_id', -1)
            anno_token = anno.get('token', gen_token(track_id, scene_token))
            anno["instance_token"] = gen_token(track_id, scene_token)
            anno['type'] = class_names_nuscenes_mappings.get(anno.get('type','unknown'), 'unknown')
            ann_dict[anno_token] = anno
        return sample_token, ann_dict

    total_annotations = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_annotation, di) for di in data_infos]
        for f in tqdm(futures, desc="Loading annotations"):
            sample_token, ann_dict = f.result()
            total_annotations[sample_token] = ann_dict

    return total_annotations

def _get_total_annotations_coop(root_path, data_infos, sample_info_mappings, max_workers=NUM_PROCESSES):

    def load_single_annotation(coop_data):
        veh_fid = coop_data['vehicle_frame']
        scene_token = sample_info_mappings[veh_fid]['scene_token']
        ann_path = osp.join(root_path, 'cooperative/label', veh_fid+'.json')
        annotations = load_json(ann_path)
        ann_dict = {}
        for anno in annotations:
            track_id = anno.get('track_id', -1)
            anno_token = anno.get('token', gen_token(track_id, scene_token))
            anno["instance_token"] = gen_token(track_id, scene_token)
            anno['type'] = class_names_nuscenes_mappings.get(anno.get('type','unknown'), 'unknown')
            ann_dict[anno_token] = anno
        return veh_fid, ann_dict

    total_annotations = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_annotation, d) for d in data_infos]
        for f in tqdm(futures, desc="Loading coop annotations"):
            veh_fid, ann_dict = f.result()
            total_annotations[veh_fid] = ann_dict

    return total_annotations

def create_spd_infos(root_path,
                     out_path,
                     v2x_side,
                     split_path,
                     can_bus_root_path,
                     info_prefix,
                     version='v1.0-trainval',
                     max_sweeps=10,
                     split_part='train',
                     flag_save=True,
                     forecasting=False,
                     forecasting_length=13,
                     max_workers=8):
    """
    Multi-process accelerated version of SPD info creation
    """
    root_path = osp.join(root_path, v2x_side)
    out_path = osp.join(out_path, v2x_side)
    data_info_path = osp.join(root_path, 'data_info.json')
    split_data_path = split_path

    data_infos = load_json(data_info_path)
    split_data = load_json(split_data_path)
    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    if v2x_side == 'vehicle-side':
        data_infos = [item for item in data_infos if item['frame_id'] not in to_remove_list_veh]

    # Generate sample mappings and annotations
    sample_infos, sample_info_mappings = _generate_sample_infos(data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations(root_path, data_infos, sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    # Get lidar -> ego -> global info
    lidar_ego_global_infos = get_lidar_ego_global_infos(root_path, data_infos, v2x_side)

    # Interpolate unvisible objects
    total_annotations = _generate_unvisible_annotations(v2x_side, sample_info_mappings, secene_frame_mappings,
                                                        instance_token_mappings, total_annotations, lidar_ego_global_infos)

    # Update instance token mappings
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    # Add velocity and prev/next
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos)

    # ---------------------------------------------------------------------
    # Define single-frame processing function inside
    # ---------------------------------------------------------------------
    def process_single_frame(data_info):
        sample_token = data_info['frame_id']
        sample_info = sample_info_mappings[sample_token]
        info = {
            'token': sample_info['token'],
            'frame_idx': sample_info['frame_idx'],
            'scene_token': sample_info['scene_token'],
            'location': sample_info['location'],
            'timestamp': sample_info['timestamp'],
            'prev': sample_info['prev'],
            'next': sample_info['next'],
        }

        # LiDAR path
        info['lidar_path'] = data_info['pointcloud_path'].replace('pcd', 'bin')
        info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
        info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
        info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
        info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

        # Camera info
        camera_type = 'VEHICLE_CAM_FRONT'
        info['cams'] = {camera_type: {}}
        info['cams'][camera_type]['data_path'] = data_info['image_path']

        key_calib_lidar2cam = 'calib_lidar_to_camera_path'
        if v2x_side == 'infrastructure-side':
            key_calib_lidar2cam = 'calib_virtuallidar_to_camera_path'
        calib_lidar2cam_path = osp.join(root_path, data_info[key_calib_lidar2cam])
        calib_lidar2cam = load_json(calib_lidar2cam_path)

        info['cams'][camera_type]['lidar2cam_rotation'] = np.array(calib_lidar2cam['rotation'])
        info['cams'][camera_type]['lidar2cam_translation'] = np.array(calib_lidar2cam['translation'])

        # sensor2lidar & sensor2ego
        cam2lidar_r = np.linalg.inv(calib_lidar2cam['rotation'])
        cam2lidar_t = -np.array(calib_lidar2cam['translation']).reshape(1, 3) @ cam2lidar_r.T
        info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
        info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(3)
        cam2ego_r, cam2ego_t = mul_matrix(cam2lidar_r, cam2lidar_t,
                                         Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                                         np.array(info['lidar2ego_translation']))
        info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
        info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)

        calib_cam_intrinsic_path = osp.join(root_path, data_info['calib_camera_intrinsic_path'])
        info['cams'][camera_type]['cam_intrinsic'] = get_cam_intr(calib_cam_intrinsic_path)

         # sweeps
        data_infos_mapping = {d['frame_id']: d for d in data_infos}
        info['sweeps'] = generate_sweeps(sample_token, sample_info_mappings, data_infos_mapping, max_sweeps=max_sweeps)

        # can_bus
        info['can_bus'] = process_can_bus(info['ego2global_translation'], info['ego2global_rotation'])


        # Annotations
        annotations = total_annotations[sample_token]
        boxes = []
        for anno_token, annotation in annotations.items():
            box3d = Box3D()
            box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'], annotation['3d_location']['z']]
            box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'], annotation['3d_dimensions']['h']]
            box3d.orientation_yaw_pitch_roll = annotation['rotation']
            box3d.name = annotation['type']
            box3d.token = annotation['token']
            box3d.instance_token = annotation['instance_token']
            box3d.track_id = int(annotation['track_id'])
            box3d.timestamp = float(sample_info['timestamp'])
            box3d.visibility = visibility_mappings[annotation['occluded_state']]
            box3d.gt_velocity = annotation['gt_velocity']
            box3d.prev = annotation['prev']
            box3d.next = annotation['next']
            boxes.append(box3d)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll for b in boxes]).reshape(-1, 1)
        info['gt_boxes'] = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        info['gt_names'] = np.array([b.name for b in boxes])
        info['gt_ins_tokens'] = np.array([b.instance_token for b in boxes])
        info['gt_inds'] = np.array([b.track_id for b in boxes])
        info['anno_tokens'] = np.array([b.token for b in boxes])
        info['timestamps'] = np.array([b.timestamp for b in boxes])
        info['visibility_tokens'] = np.array([b.visibility for b in boxes])
        info['gt_velocity'] = np.array([b.gt_velocity for b in boxes])
        info['prev_anno_tokens'] = np.array([b.prev for b in boxes])
        info['next_anno_tokens'] = np.array([b.next for b in boxes])
        info['valid_flag'] = np.array([True for _ in boxes])
        info['num_lidar_pts'] = np.array([1 for _ in boxes])

        # Forecasting
        if forecasting:
            fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(
                instance_token_mappings, lidar_ego_global_infos, annotations, forecasting_length)
            info['forecasting_locs'] = np.array([np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in fboxes])
            info['forecasting_tokens'] = np.array([np.array([b.token for b in boxes]) for boxes in fboxes])
            info['forecasting_masks'] = np.array(fmasks)
            info['forecasting_types'] = np.array(ftypes)

        return info
    # ---------------------------------------------------------------------
    # Multi-process execution
    # ---------------------------------------------------------------------
    spd_infos = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_frame, data_info) for data_info in data_infos]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            spd_infos.append(fut.result())

    # Split train / val
    train_spd_infos = [info for info in spd_infos if sample_info_mappings[info['token']]['scene_token'] in train_scenes]
    val_spd_infos = [info for info in spd_infos if sample_info_mappings[info['token']]['scene_token'] in val_scenes]

    # Save
    if flag_save:
        metadata = dict(version=version)
        mmcv.dump(dict(infos=train_spd_infos, metadata=metadata),
                   osp.join(out_path, f'{info_prefix}_infos_temporal_train.pkl'))
        mmcv.dump(dict(infos=val_spd_infos, metadata=metadata),
                   osp.join(out_path, f'{info_prefix}_infos_temporal_val.pkl'))

    return total_annotations, sample_info_mappings, spd_infos

def create_spd_infos_coop(root_path,
                             out_path,
                             v2x_side,
                             split_path,
                             can_bus_root_path,
                             info_prefix,
                             version='v1.0-trainval',
                             max_sweeps=10,
                             split_part='train',
                             flag_save=True,
                             forecasting=False,
                             forecasting_length=13,
                             max_workers=8):
    """
    Multi-process version of create_spd_infos_coop.

    Args:
        root_path: dataset root
        out_path: output path
        v2x_side: 'cooperative'
        split_path: train/val split json
        can_bus_root_path: optional
        info_prefix: info file prefix
        forecasting: whether to include future trajectory
        forecasting_length: length of forecasting
        max_workers: number of parallel processes
    """

    out_path = osp.join(out_path, v2x_side)

    # Step 0: load data
    coop_data_info_path = osp.join(root_path, 'cooperative/data_info.json')
    coop_data_infos = load_json(coop_data_info_path)
    coop_data_infos = [item for item in coop_data_infos if item['vehicle_frame'] not in to_remove_list_coop]

    veh_data_info_path = osp.join(root_path, 'vehicle-side/data_info.json')
    veh_data_infos = load_json(veh_data_info_path)

    inf_data_info_path = osp.join(root_path, 'infrastructure-side/data_info.json')
    inf_data_infos = load_json(inf_data_info_path)

    split_data = load_json(split_path)
    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    # Step 1: generate mappings
    data_infos_mapping_veh = {d['frame_id']: d for d in veh_data_infos}
    data_infos_mapping_inf = {d['frame_id']: d for d in inf_data_infos}
    sample_infos, sample_info_mappings = _generate_sample_infos_coop(coop_data_infos, veh_data_infos, inf_data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations_coop(root_path, coop_data_infos, sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    # Step 2: lidar->ego->global info
    lidar_ego_global_infos = get_lidar_ego_global_infos(osp.join(root_path, 'vehicle-side'), veh_data_infos, v2x_side)

    # Step 3: interpolate unvisible boxes
    total_annotations = _generate_unvisible_annotations("cooperative", sample_info_mappings, secene_frame_mappings,
                                                        instance_token_mappings, total_annotations, lidar_ego_global_infos)

    # Step 4: update instance mappings
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    # Step 5: add velocity & prev/next
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(
        total_annotations, instance_token_mappings, lidar_ego_global_infos
    )

    # ---------------------------------------------------------------------
    # Step 6: define single-frame processing
    # ---------------------------------------------------------------------
    def process_single_frame(coop_data_info):
        veh_frame_id = coop_data_info['vehicle_frame']
        inf_frame_id = coop_data_info['infrastructure_frame']
        sample_token = veh_frame_id
        sample_info = sample_info_mappings[sample_token]

        info = {
            'token': sample_info['token'],
            'frame_idx': sample_info['frame_idx'],
            'scene_token': sample_info['scene_token'],
            'location': sample_info['location'],
            'timestamp': sample_info['timestamp'],
            'prev': sample_info['prev'],
            'next': sample_info['next'],
            'token_inf': sample_info['token_inf'],
            'timestamp_inf': sample_info['timestamp_inf'],
            'system_error_offset': sample_info['system_error_offset']
        }

        # Vehicle info
        veh_frame_infos, veh_info_mappings = _generate_sample_infos(veh_data_infos)
        veh_data_info = veh_info_mappings[veh_frame_id]
        info['lidar_path'] = veh_data_info['pointcloud_path'].replace('pcd', 'bin')
        info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
        info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
        info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
        info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

        # Infrastructure info
        inf_frame_infos, inf_info_mappings = _generate_sample_infos(inf_data_infos)
        inf_data_info = inf_info_mappings[inf_frame_id]
        info['lidar2ego_rotation_inf'] = np.eye(3)
        info['lidar2ego_translation_inf'] = np.zeros(3)
        calib_virtuallidar2global_path = osp.join(root_path, 'infrastructure-side', inf_data_info['calib_virtuallidar_to_world_path'])
        calib_virtuallidar2global = load_json(calib_virtuallidar2global_path)
        info['ego2global_rotation_inf'] = np.array(calib_virtuallidar2global['rotation'])
        info['ego2global_translation_inf'] = np.array(calib_virtuallidar2global['translation']).reshape(3)

        # Coop transformation: VehLidar -> InfLidar
        veh_l2e_r = np.array(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
        veh_l2e_t = np.array(info['lidar2ego_translation']).reshape(3)
        veh_e2g_r = np.array(Quaternion(info['ego2global_rotation']).rotation_matrix)
        veh_e2g_t = np.array(info['ego2global_translation']).reshape(3)

        inf_e2g_r = np.array(calib_virtuallidar2global['rotation'])
        inf_e2g_t = np.array(calib_virtuallidar2global['translation']).reshape(3)
        inf_l2e_r = np.eye(3)
        inf_l2e_t = np.zeros(3)

        err_offset = np.array([sample_info['system_error_offset']['delta_x'],
                               sample_info['system_error_offset']['delta_y'], 0])
        r = ((veh_l2e_r.T @ veh_e2g_r.T) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
        t = (-err_offset @ veh_l2e_r.T @ veh_e2g_r.T + veh_l2e_t @ veh_e2g_r.T + veh_e2g_t) @ \
            (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
        t -= inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) + inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T)
        info['VehLidar2InfLidar_rotation'] = r
        info['VehLidar2InfLidar_translation'] = t

        # Cameras, sweeps, can_bus placeholders
        info['cams'] = {}
        camera_type = 'VEHICLE_CAM_FRONT'
        info['cams'][camera_type] = {}
        info['cams'][camera_type]['data_path'] = veh_data_info['image_path']

        calib_lidar2cam_path = osp.join(root_path, 'vehicle-side', veh_data_info['calib_lidar_to_camera_path'])
        calib_lidar2cam = load_json(calib_lidar2cam_path)
        info['cams'][camera_type]['lidar2cam_rotation'] = np.array(calib_lidar2cam['rotation'])
        info['cams'][camera_type]['lidar2cam_translation'] = np.array(calib_lidar2cam['translation'])

        cam2lidar_r = np.linalg.inv(calib_lidar2cam['rotation'])
        cam2lidar_t = - np.array(calib_lidar2cam['translation']).reshape(1, 3) @ cam2lidar_r.T
        cam2ego_r, cam2ego_t = mul_matrix(cam2lidar_r, cam2lidar_t,
                                          Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                                          np.array(info['lidar2ego_translation']))
        info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
        info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(3)
        info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
        info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)

        calib_cam_intrinsic_path = osp.join(root_path, 'vehicle-side', veh_data_info['calib_camera_intrinsic_path'])
        info['cams'][camera_type]['cam_intrinsic'] = get_cam_intr(calib_cam_intrinsic_path)

        # info['sweeps'] = {}
        info['sweeps_inf'] = generate_sweeps(
            sample_token=inf_frame_id,
            sample_info_mappings=sample_info_mappings,
            data_infos_mapping=data_infos_mapping_inf,
            max_sweeps=max_sweeps
        )
        info['sweeps_inf'] = generate_sweeps(
                            sample_token=inf_frame_id,
                            sample_info_mappings=sample_info_mappings,
                            data_infos_mapping=data_infos_mapping_inf,
        max_sweeps=max_sweeps
    )
        # info['can_bus'] = np.zeros(18)
        info['can_bus'] = process_can_bus(info['ego2global_translation'], info['ego2global_rotation'])

        # Annotations
        annotations = total_annotations[sample_token]
        boxes = []
        for anno_token, annotation in annotations.items():
            box3d = Box3D()
            box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'], annotation['3d_location']['z']]
            box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'], annotation['3d_dimensions']['h']]
            box3d.orientation_yaw_pitch_roll = annotation['rotation']
            box3d.name = annotation['type']
            box3d.token = annotation['token']
            box3d.instance_token = annotation['instance_token']
            box3d.track_id = int(annotation['track_id'])
            box3d.timestamp = float(sample_info['timestamp'])
            box3d.visibility = visibility_mappings[annotation['occluded_state']]
            box3d.gt_velocity = annotation['gt_velocity']
            box3d.prev = annotation['prev']
            box3d.next = annotation['next']
            boxes.append(box3d)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll for b in boxes]).reshape(-1, 1)
        info['gt_boxes'] = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        info['gt_names'] = np.array([b.name for b in boxes])
        info['gt_ins_tokens'] = np.array([b.instance_token for b in boxes])
        info['gt_inds'] = np.array([b.track_id for b in boxes])
        info['anno_tokens'] = np.array([b.token for b in boxes])
        info['timestamps'] = np.array([b.timestamp for b in boxes])
        info['visibility_tokens'] = np.array([b.visibility for b in boxes])
        info['gt_velocity'] = np.array([b.gt_velocity for b in boxes])
        info['prev_anno_tokens'] = np.array([b.prev for b in boxes])
        info['next_anno_tokens'] = np.array([b.next for b in boxes])
        info['valid_flag'] = np.array([True for _ in boxes])
        info['num_lidar_pts'] = np.array([1 for _ in boxes])

        # Forecasting
        if forecasting:
            fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(
                instance_token_mappings, lidar_ego_global_infos, annotations, forecasting_length)
            locs_f = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in fboxes]
            tokens_f = [np.array([b.token for b in boxes]) for boxes in fboxes]
            info['forecasting_locs'] = np.array(locs_f)
            info['forecasting_tokens'] = np.array(tokens_f)
            info['forecasting_masks'] = np.array(fmasks)
            info['forecasting_types'] = np.array(ftypes)

        return info

    # ---------------------------------------------------------------------
    # Step 7: multi-process execution
    # ---------------------------------------------------------------------
    spd_infos = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_frame, data_info) for data_info in coop_data_infos]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            spd_infos.append(fut.result())

    # Split train/val
    train_spd_infos = [info for info in spd_infos if sample_info_mappings[info['token']]['scene_token'] in train_scenes]
    val_spd_infos = [info for info in spd_infos if sample_info_mappings[info['token']]['scene_token'] in val_scenes]

    # Save
    if flag_save:
        metadata = dict(version=version)
        mmcv.dump(dict(infos=train_spd_infos, metadata=metadata),
                   osp.join(out_path, f'{info_prefix}_infos_temporal_train.pkl'))
        mmcv.dump(dict(infos=val_spd_infos, metadata=metadata),
                   osp.join(out_path, f'{info_prefix}_infos_temporal_val.pkl'))

    return total_annotations, sample_info_mappings, spd_infos

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--save-root', type=str, default="./data/infos/V2X-Seq-SPD-Example")
    parser.add_argument('--split-file', type=str, default="./data/split_datas/cooperative-split-data-spd.json")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="spd")
    parser.add_argument('--forecasting', action='store_true', default=False, help='prepare trajectory forecasting data')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    v2x_side = args.v2x_side
    data_root = args.data_root
    save_root = args.save_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix
    forecasting = args.forecasting

    print(data_root)
    print(save_root)
    print(v2x_side)
    if forecasting:
        print('prepare trajectory forecasting data')
    else:
        print('not prepare trajectory forecasting data')

    if v2x_side == 'cooperative':
        # generate_json_maps_files(data_root, version=v2x_side)
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos_coop(
                            data_root,
                            save_root,
                            v2x_side,
                            split_path, 
                            can_bus_root_path,
                            info_prefix,
                            version=args.version,
                            max_sweeps=10,
                            forecasting=forecasting)     
    else:   
        # generate_json_maps_files(data_root, version=v2x_side)
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos(
                                                                            data_root,
                                                                            save_root,
                                                                            v2x_side,
                                                                            split_path,
                                                                            can_bus_root_path,
                                                                            info_prefix,
                                                                            version=args.version,
                                                                            max_sweeps=10,
                                                                            forecasting=forecasting)

