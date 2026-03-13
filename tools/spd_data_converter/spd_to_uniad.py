from pickle import NONE
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
import struct

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
import os

# GlOBAL

TOTAL_CPU = os.cpu_count()
RESERVE_CPU = 2  # 系统保留
MAX_AVAILABLE = max(1, TOTAL_CPU - RESERVE_CPU)

# 进程池：每个 worker 独立加载点云，内存占用大。限制数量减少 OOM。
PROCESS_POOL_MAX_WORKERS = min(4, MAX_AVAILABLE)

# 线程池：共享主进程内存。I/O 类可多用，CPU 类适度即可。
THREAD_POOL_IO_WORKERS = min(12, MAX_AVAILABLE)   # 读 JSON 等 I/O
THREAD_POOL_CPU_WORKERS = min(8, MAX_AVAILABLE)   # annotation 转换等 CPU 密集
# create_spd_infos: ProcessPool 并行（绕过 GIL），worker 数可提高
CREATE_SPD_INFOS_WORKERS = min(8, MAX_AVAILABLE)

# 兼容旧逻辑的别名
NUM_PROCESSES = PROCESS_POOL_MAX_WORKERS
WORKERS_PER_PROCESS = max(1, THREAD_POOL_CPU_WORKERS)

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

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'van']
PEDESTRIAN_CLASSES = ['pedestrian']
CYCLE_CLASSES = ['bicycle', 'motorcycle']

MOVING_THRESH = 0.2  # m/s
STATIC_THRESH = 0.05

ATTRIBUTE_MAPPING = {
    'vehicle.moving': 0,
    'vehicle.parked': 1,
    'vehicle.stopped': 2,
    'pedestrian.moving': 3,
    'pedestrian.standing': 4,
    'cycle.with_rider': 5,
}


def orthonormalize_rotation(R):
    """
    Fix non-orthogonal rotation matrix using SVD. Returns proper rotation (det=+1).
    Ensures matrix is valid for pyquaternion.Quaternion(matrix=...).
    """
    R = np.array(R, dtype=np.float64)
    U, S, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    # Ensure proper rotation (det=+1), not reflection (det=-1)
    if np.linalg.det(R_orth) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R_orth = U @ Vt
    return R_orth.astype(np.float32)


def iterative_closest_point(A, num_iterations=100):
    """Polar decomposition to get nearest orthogonal matrix (for coop inf calib)."""
    R = np.array(A, dtype=np.float64).copy()
    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U
    return R


def get_single_sample_info(frame_id, data_infos):
    """Get data_info dict for a given frame_id."""
    for d in data_infos:
        if d['frame_id'] == frame_id:
            return d
    raise KeyError(f"frame_id {frame_id} not found in data_infos")


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


def _lzf_decompress(data, expected_length):
    """Minimal LZF decompressor for PCD binary_compressed blocks."""
    i = 0
    o = 0
    out = bytearray(expected_length if expected_length is not None else 0)
    data_len = len(data)
    while i < data_len:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            length = ctrl + 1
            if expected_length is not None and o + length > expected_length:
                raise ValueError("LZF literal overrun.")
            if expected_length is None:
                out.extend(data[i:i + length])
            else:
                out[o:o + length] = data[i:i + length]
            o += length
            i += length
        else:
            length = ctrl >> 5
            ref = o - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += data[i]
                i += 1
            ref -= data[i]
            i += 1
            length += 2
            if ref < 0:
                raise ValueError("Invalid LZF back-reference.")
            if expected_length is not None and o + length > expected_length:
                raise ValueError("LZF back-reference overrun.")
            for _ in range(length):
                if expected_length is None:
                    out.append(out[ref])
                else:
                    out[o] = out[ref]
                o += 1
                ref += 1
    if expected_length is not None and o != expected_length:
        raise ValueError(
            f"LZF decompressed size mismatch: got {o}, expected {expected_length}")
    return bytes(out if expected_length is None else out[:o])


def _load_points_xyz_from_pcd(pcd_path):
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PCD: missing DATA header.")
            line_decoded = line.decode('ascii', errors='ignore').strip()
            if not line_decoded or line_decoded.startswith('#'):
                continue
            parts = line_decoded.split()
            key = parts[0].upper()
            value = parts[1:]
            header[key] = value
            if key == 'DATA':
                break
        data_blob = f.read()

    fields = header.get('FIELDS', [])
    sizes = list(map(int, header.get('SIZE', [])))
    types = header.get('TYPE', [])
    counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
    points = int(header.get('POINTS', [header.get('WIDTH', ['0'])[0]])[0])
    data_type = header.get('DATA', ['binary'])[0].lower()
    if not fields or not sizes or not types or len(fields) != len(sizes) or len(fields) != len(types):
        raise ValueError("Invalid PCD header fields.")

    bytes_per_field = [s * c for s, c in zip(sizes, counts)]
    xyz_idx = [fields.index('x'), fields.index('y'), fields.index('z')]

    def _dtype_of(size, typ):
        if typ == 'F' and size == 4:
            return np.float32
        if typ == 'F' and size == 8:
            return np.float64
        if typ == 'U' and size == 1:
            return np.uint8
        if typ == 'U' and size == 2:
            return np.uint16
        if typ == 'U' and size == 4:
            return np.uint32
        if typ == 'I' and size == 1:
            return np.int8
        if typ == 'I' and size == 2:
            return np.int16
        if typ == 'I' and size == 4:
            return np.int32
        raise ValueError(f"Unsupported PCD type/size: {typ}{size}")

    if data_type == 'ascii':
        text = data_blob.decode('ascii', errors='ignore').strip()
        if not text:
            return np.zeros((0, 3), dtype=np.float32)
        data = np.loadtxt(text.splitlines(), dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        return data[:, xyz_idx].astype(np.float32)

    if data_type == 'binary':
        point_step = sum(bytes_per_field)
        expected = points * point_step
        raw = data_blob[:expected]
        if len(raw) < expected:
            raise ValueError("Truncated PCD binary payload.")
        offsets = []
        cur = 0
        for field_len in bytes_per_field:
            offsets.append(cur)
            cur += field_len
        dtype_desc = []
        for name, size, typ, cnt, off in zip(fields, sizes, types, counts, offsets):
            dt = _dtype_of(size, typ)
            shape = (cnt,) if cnt > 1 else ()
            dtype_desc.append((name, dt, shape, off))
        structured_dtype = np.dtype(
            {'names': [x[0] for x in dtype_desc],
             'formats': [np.dtype((x[1], x[2])) for x in dtype_desc],
             'offsets': [x[3] for x in dtype_desc],
             'itemsize': point_step}
        )
        arr = np.frombuffer(raw, dtype=structured_dtype, count=points)
        return np.stack(
            [arr['x'].reshape(points, -1)[:, 0],
             arr['y'].reshape(points, -1)[:, 0],
             arr['z'].reshape(points, -1)[:, 0]],
            axis=1
        ).astype(np.float32)

    if data_type == 'binary_compressed':
        if len(data_blob) < 8:
            raise ValueError("Invalid compressed PCD payload.")
        compressed_size = struct.unpack('<I', data_blob[:4])[0]
        uncompressed_size = struct.unpack('<I', data_blob[4:8])[0]
        compressed = data_blob[8:8 + compressed_size]
        raw = _lzf_decompress(compressed, uncompressed_size)

        # PCD compressed payload is stored in structure-of-arrays layout.
        xyz = {}
        offset = 0
        for i, (size, typ, cnt) in enumerate(zip(sizes, types, counts)):
            dt = _dtype_of(size, typ)
            field_bytes = points * size * cnt
            block = raw[offset:offset + field_bytes]
            if i in xyz_idx:
                arr = np.frombuffer(block, dtype=dt, count=points * cnt)
                arr = arr.reshape(points, cnt)[:, 0]
                xyz[fields[i]] = arr.astype(np.float32)
            offset += field_bytes
        return np.stack([xyz['x'], xyz['y'], xyz['z']], axis=1)

    raise ValueError(f"Unsupported PCD DATA type: {data_type}")


def _load_points_xyz(lidar_path, load_dim=4):
    # Prefer explicit path; if .bin is missing, fall back to matching .pcd.
    candidates = [lidar_path]
    if lidar_path.endswith('.bin'):
        candidates.append(lidar_path[:-4] + '.pcd')
    for p in candidates:
        if not osp.isfile(p):
            continue
        if p.endswith('.bin'):
            points = np.fromfile(p, dtype=np.float32).reshape(-1, load_dim)
            return points[:, :3]
        if p.endswith('.pcd'):
            return _load_points_xyz_from_pcd(p)
    raise FileNotFoundError(lidar_path)


def compute_num_lidar_pts_per_box(lidar_path, gt_boxes, load_dim=4):
    """Compute number of lidar points inside each 3D box (for valid_flag / filtering).
    gt_boxes: (N, 7) in lidar frame, format (x,y,z,w,l,h,yaw) with yaw as in SECOND (-rot - pi/2).
    Returns: (N,) int array of point counts.
    """
    try:
        pts_xyz = _load_points_xyz(lidar_path, load_dim=load_dim)
    except Exception:
        # Fallback keeps pipeline robust when source point cloud is missing.
        return np.ones(len(gt_boxes), dtype=np.int64)
    if len(gt_boxes) == 0:
        return np.array([], dtype=np.int64)
    num_pts = np.zeros(len(gt_boxes), dtype=np.int64)
    cx, cy, cz = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
    w, l, h = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
    yaw = gt_boxes[:, 6]
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    dx = pts_xyz[:, 0] - cx[:, None]  # (N_boxes, N_pts)
    dy = pts_xyz[:, 1] - cy[:, None]
    dz = pts_xyz[:, 2] - cz[:, None]
    local_x = cos_yaw[:, None] * dx + sin_yaw[:, None] * dy
    local_y = -sin_yaw[:, None] * dx + cos_yaw[:, None] * dy
    local_z = dz
    in_x = (np.abs(local_x) <= (l / 2)[:, None])
    in_y = (np.abs(local_y) <= (w / 2)[:, None])
    in_z = (np.abs(local_z) <= (h / 2)[:, None])
    inside = in_x & in_y & in_z  # (N_boxes, N_pts)
    num_pts[:] = inside.sum(axis=1)
    return num_pts


def points_in_boxes_mask(pts_xyz, gt_boxes):
    """pts_xyz (N, 3), gt_boxes (K, 7). Returns (K, N) bool."""
    if len(gt_boxes) == 0:
        return np.zeros((0, pts_xyz.shape[0]), dtype=bool)
    cx, cy, cz = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
    w, l, h = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
    yaw = gt_boxes[:, 6]
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    dx = pts_xyz[:, 0] - cx[:, None]
    dy = pts_xyz[:, 1] - cy[:, None]
    dz = pts_xyz[:, 2] - cz[:, None]
    local_x = cos_yaw[:, None] * dx + sin_yaw[:, None] * dy
    local_y = -sin_yaw[:, None] * dx + cos_yaw[:, None] * dy
    local_z = dz
    in_x = np.abs(local_x) <= (l / 2)[:, None]
    in_y = np.abs(local_y) <= (w / 2)[:, None]
    in_z = np.abs(local_z) <= (h / 2)[:, None]
    return in_x & in_y & in_z

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

# UniV2X TODO: remapping
# UniV2X TODO: modify the related code in UniAD and nuScenes
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


# ---------Core base DataProcess Function-----

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


def infer_nuscenes_attribute(
    name,
    velocity,
    visibility=None,
    valid_flag=True,
    speed_thresh=MOVING_THRESH
):
    """
    Infer nuScenes-style attribute from SPD annotations.

    Args:
        name (str): class name
        velocity (array-like): [vx, vy] or [vx, vy, vz]
        visibility (int): visibility level (optional)
        valid_flag (bool): whether box is valid (has lidar pts)
    Returns:
        attr_name (str or None)
    """
    if not valid_flag:
        return None

    if velocity is None or len(velocity) < 2:
        speed = 0.0
    else:
        speed = np.linalg.norm(velocity)

    # Vehicle
    if name in VEHICLE_CLASSES:
        if speed > speed_thresh:
            return 'vehicle.moving'
        else:
            # static vehicle
            if visibility is not None and visibility < 2:
                return 'vehicle.parked'
            else:
                return 'vehicle.stopped'

    # Pedestrian
    if name in PEDESTRIAN_CLASSES:
        if speed > speed_thresh:
            return 'pedestrian.moving'
        else:
            return 'pedestrian.standing'

    # Cycle
    if name in CYCLE_CLASSES:
        return 'cycle.with_rider'

    return None


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
            # lidar -> ego 默认 identity，用单位四元数表示，方便后续构造 Quaternion
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.zeros(
                3, dtype=np.float32)

            # ego -> global
            calib_path = osp.join(
                root_path, data_info['calib_virtuallidar_to_world_path'])
            calib = load_json(calib_path)

            R = np.array(calib['rotation'], dtype=np.float32).reshape(3, 3)
            t = np.array(calib['translation'], dtype=np.float32).reshape(3)

            R = orthonormalize_rotation(R)
            try:
                q = Quaternion(matrix=R)
            except ValueError:
                R = iterative_closest_point(R.astype(np.float64))
                q = Quaternion(matrix=R)

            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(
                [q.w, q.x, q.y, q.z], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = t

        else:  # vehicle-side（与 vis_gt 一致：直接使用标定矩阵，不做 orthonormalize，保证 velocity 等计算一致）
            calib_l2e_path = osp.join(
                root_path, data_info['calib_lidar_to_novatel_path'])
            calib_l2e = load_json(calib_l2e_path)
            q_l2e = Quaternion(matrix=np.array(
                calib_l2e['transform']['rotation']))
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(
                [q_l2e.w, q_l2e.x, q_l2e.y, q_l2e.z], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array(
                calib_l2e['transform']['translation'], dtype=np.float32).reshape(3)

            calib_e2g_path = osp.join(
                root_path, data_info['calib_novatel_to_world_path'])
            calib_e2g = load_json(calib_e2g_path)
            q_e2g = Quaternion(matrix=np.array(calib_e2g['rotation']))
            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(
                [q_e2g.w, q_e2g.x, q_e2g.y, q_e2g.z], dtype=np.float32)
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(
                calib_e2g['translation'], dtype=np.float32).reshape(3)

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
        cur_time = float(sample_info_mappings[token]['timestamp'])

        # ---------- 第一帧 ----------
        if i == 0:
            next_token = data_infos[i+1]['frame_id']
            next_loc = np.asarray(
                lidar_ego_global_infos[next_token]['ego2global_translation'],
                dtype=np.float32
            )
            next_time = float(
                sample_info_mappings[next_token]['timestamp'])
            dt = next_time - cur_time
            ego_velocity[token] = (next_loc - cur_loc)[:2] / max(dt, 1e-6)

        # ---------- 最后一帧 ----------
        elif i == n - 1:
            prev_token = data_infos[i-1]['frame_id']
            prev_loc = np.asarray(
                lidar_ego_global_infos[prev_token]['ego2global_translation'],
                dtype=np.float32
            )
            prev_time = float(
                sample_info_mappings[prev_token]['timestamp'])
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

            prev_time = float(
                sample_info_mappings[prev_token]['timestamp'])
            next_time = float(
                sample_info_mappings[next_token]['timestamp'])

            dt = next_time - prev_time
            ego_velocity[token] = (next_loc - prev_loc)[:2] / max(dt, 1e-6)

    return ego_velocity


def _build_sweep_data_mapping(data_infos, lidar_ego_global_infos):
    """Build {frame_id: sweep_info} with lidar_path, timestamp, lidar2ego, ego2global."""
    mapping = {}
    for d in data_infos:
        fid = d['frame_id']
        if fid not in lidar_ego_global_infos:
            continue
        ts = float(d.get('pointcloud_timestamp', d.get('timestamp', 0)))
        # keep timestamp: seconds if input was us, else as-is
        ts_sec = ts / 1e6 if ts > 1e10 else ts
        mapping[fid] = {
            'lidar_path': d.get('pointcloud_path', d.get('lidar_path', '')),
            'timestamp': ts_sec,
            # microseconds for mmdet3d
            'timestamp_us': int(ts) if ts > 1e10 else int(ts * 1e6),
            'lidar2ego_rotation': lidar_ego_global_infos[fid]['lidar2ego_rotation'],
            'lidar2ego_translation': lidar_ego_global_infos[fid]['lidar2ego_translation'],
            'ego2global_rotation': lidar_ego_global_infos[fid]['ego2global_rotation'],
            'ego2global_translation': lidar_ego_global_infos[fid]['ego2global_translation'],
        }
    return mapping


def generate_sweeps(sample_token, sample_info_mappings, data_infos_mapping,
                    lidar_ego_global_infos, max_sweeps=10):
    """Generate sweeps in mmdet3d format: data_path, timestamp (us), sensor2lidar_rotation, sensor2lidar_translation."""
    sweeps = []
    cur_timestamp = float(sample_info_mappings[sample_token]['timestamp'])
    cur_ts_us = int(cur_timestamp) if cur_timestamp > 1e10 else int(
        cur_timestamp * 1e6)
    prev_token = sample_info_mappings[sample_token]['prev']

    if sample_token not in lidar_ego_global_infos:
        return sweeps
    kf_l2e_r = Quaternion(
        lidar_ego_global_infos[sample_token]['lidar2ego_rotation']).rotation_matrix
    kf_l2e_t = np.array(
        lidar_ego_global_infos[sample_token]['lidar2ego_translation'])
    kf_e2g_r = Quaternion(
        lidar_ego_global_infos[sample_token]['ego2global_rotation']).rotation_matrix
    kf_e2g_t = np.array(
        lidar_ego_global_infos[sample_token]['ego2global_translation'])

    while len(sweeps) < max_sweeps and prev_token != '' and prev_token in data_infos_mapping:
        sweep_info = data_infos_mapping[prev_token]
        l2e_r_s = Quaternion(sweep_info['lidar2ego_rotation']).rotation_matrix
        l2e_t_s = np.array(sweep_info['lidar2ego_translation'])
        e2g_r_s = Quaternion(sweep_info['ego2global_rotation']).rotation_matrix
        e2g_t_s = np.array(sweep_info['ego2global_translation'])

        # sweep lidar -> keyframe lidar (same as obtain_sensor2top in uniad_nuscenes_converter)
        R = (l2e_r_s.T @ e2g_r_s.T) @ (np.linalg.inv(kf_e2g_r).T @
                                       np.linalg.inv(kf_l2e_r).T)
        T = (l2e_t_s @ e2g_r_s.T + e2g_t_s) @ (np.linalg.inv(kf_e2g_r).T @
                                               np.linalg.inv(kf_l2e_r).T)
        T -= kf_e2g_t @ (np.linalg.inv(kf_e2g_r).T @
                         np.linalg.inv(kf_l2e_r).T) + kf_l2e_t @ np.linalg.inv(kf_l2e_r).T

        ts_us = sweep_info.get('timestamp_us')
        if ts_us is None:
            ts_sec = sweep_info['timestamp']
            ts_us = int(ts_sec * 1e6) if ts_sec < 1e10 else int(ts_sec)

        sweeps.append({
            'lidar_path': sweep_info['lidar_path'],
            'data_path': sweep_info['lidar_path'],
            'timestamp': ts_us,
            'sensor2lidar_rotation': R.T,
            'sensor2lidar_translation': T,
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
    与 vis_gt 完全一致：timestamp /1e6 转秒，center 在 global 下插值
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    # cvt to global
    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['lidar2ego_rotation']).rotation_matrix, center_0) \
        + np.array(lidar_ego_global_info_0['lidar2ego_translation'])
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['ego2global_rotation']).rotation_matrix, center_0) \
        + np.array(lidar_ego_global_info_0['ego2global_translation'])

    # cvt to global
    center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['lidar2ego_rotation']).rotation_matrix, center_1) \
        + np.array(lidar_ego_global_info_1['lidar2ego_translation'])
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['ego2global_rotation']).rotation_matrix, center_1)\
        + np.array(lidar_ego_global_info_1['ego2global_translation'])

    # global interpolation
    ratio = (cur_timestamp - timestamp_ii_0)/(timestamp_ii_1 - timestamp_ii_0)
    cur_center = (1 - ratio) * center_0 + ratio * center_1

    # cur sesor data interpolation
    global2ego_r = np.linalg.inv(Quaternion(
        cur_lidar_ego_global_info['ego2global_rotation']).rotation_matrix)
    global2ego_t = - \
        np.array(cur_lidar_ego_global_info['ego2global_translation']).reshape(
            1, 3) @ global2ego_r.T
    global2ego_t = global2ego_t.reshape(3)

    ego2lidar_r = np.linalg.inv(Quaternion(
        cur_lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix)
    ego2lidar_t = - \
        np.array(cur_lidar_ego_global_info['lidar2ego_translation']).reshape(
            1, 3) @ ego2lidar_r.T
    ego2lidar_t = ego2lidar_t.reshape(3)

    cur_center = np.dot(global2ego_r, cur_center) + global2ego_t
    cur_center = np.dot(ego2lidar_r, cur_center) + ego2lidar_t
    locs_out = {'x': cur_center[0], 'y': cur_center[1], 'z': cur_center[2]}

    return locs_out


def rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp):
    """Use linear interpolation to estimate the rotation for occluded objects.
    与 vis_gt 一致
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    time_ratio = (cur_timestamp - timestamp_ii_0) / \
        (timestamp_ii_1 - timestamp_ii_0)
    diff = rot_ii_1 - rot_ii_0 + pi
    if diff < 0:
        diff = diff + pi
    elif diff > 2 * pi:
        diff = diff - 3 * pi
    else:
        diff = diff - pi

    cur_rot = time_ratio * diff + rot_ii_0
    if cur_rot < -pi:
        cur_rot = cur_rot + 2 * pi
    if cur_rot > 2 * pi:
        cur_rot = cur_rot - 2 * pi

    return cur_rot

# -----------Veloctity Process--------------


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
    def process_instance(instance_token):
        results = []
        cur_instance_samples = instance_token_mappings[instance_token]

        for ii in range(len(cur_instance_samples) - 1):
            cur_frame_idx = cur_instance_samples[ii]['frame_idx'] + 1
            while cur_frame_idx != cur_instance_samples[ii + 1]['frame_idx']:
                # 插值所需参数
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii +
                                                1]['annotation']['3d_location']
                rot_ii_0 = cur_instance_samples[ii]['annotation']['rotation']
                rot_ii_1 = cur_instance_samples[ii +
                                                1]['annotation']['rotation']
                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']

                cur_sample_token = secene_frame_mappings[
                    (cur_instance_samples[0]['scene_token'], cur_frame_idx)
                ]

                # 插值位置和旋转
                cur_loc = loc_linear_interpolation(
                    loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1,
                    sample_info_mappings[cur_sample_token]['timestamp'],
                    lidar_ego_global_infos[cur_instance_samples[ii]
                                           ['sample_token']],
                    lidar_ego_global_infos[cur_instance_samples[ii+1]
                                           ['sample_token']],
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
                results.append(
                    (cur_sample_token, cur_anno_token, cur_instance_sample_anno))
                cur_frame_idx += 1
        return results

    # 保持与 vis_gt 一致的迭代顺序（instance_token_mappings 顺序），确保 total_annotations 确定性
    for instance_token in tqdm(instance_token_mappings, desc="annotation_track"):
        for sample_token, anno_token, anno in process_instance(instance_token):
            if sample_token not in total_annotations:
                total_annotations[sample_token] = {}
            total_annotations[sample_token][anno_token] = anno

    return total_annotations


def global_to_lidar(global_point, lidar_ego_global_info):
    """
    Convert global coordinate to lidar coordinate.

    Args:
        global_point (np.array): [x, y, z] in global coordinate
        lidar_ego_global_info (dict): lidar->ego and ego->global transforms

    Returns:
        np.array: [x, y, z] in lidar coordinate
    """
    # Get transformation matrices
    e2g_r = Quaternion(
        lidar_ego_global_info['ego2global_rotation']).rotation_matrix
    e2g_t = np.array(lidar_ego_global_info['ego2global_translation'])
    l2e_r = Quaternion(
        lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix
    l2e_t = np.array(lidar_ego_global_info['lidar2ego_translation'])

    # Global -> Ego
    ego_point = np.dot(e2g_r.T, global_point - e2g_t)

    # Ego -> LiDAR
    lidar_point = np.dot(l2e_r.T, ego_point - l2e_t)

    return lidar_point


def _quaternion_yaw(q):
    """Extract yaw (rad) from quaternion. Same logic as in process_can_bus."""
    w, x, y, z = q.elements
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def global_yaw_to_lidar(yaw_global_rad, lidar_ego_global_info):
    """
    Convert box yaw from global to lidar frame.
    yaw_lidar = yaw_global - ego_yaw, wrapped to [-pi, pi]
    """
    ego_yaw = _quaternion_yaw(Quaternion(lidar_ego_global_info['ego2global_rotation']))
    yaw_lidar = yaw_global_rad - ego_yaw
    yaw_lidar = (yaw_lidar + np.pi) % (2 * np.pi) - np.pi
    return yaw_lidar


def global_velocity_to_lidar(velocity_xy_global, lidar_ego_global_info):
    """
    Convert velocity from global frame to lidar frame (NuScenes convention).
    NuScenes stores gt_velocity in lidar frame (absolute velocity expressed in lidar axes).

    Args:
        velocity_xy_global (np.array): [vx, vy] in global frame (m/s)
        lidar_ego_global_info (dict): lidar->ego and ego->global transforms

    Returns:
        np.array: [vx, vy] in lidar frame
    """
    e2g_r = Quaternion(
        lidar_ego_global_info['ego2global_rotation']).rotation_matrix
    l2e_r = Quaternion(
        lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix
    velo = np.array(
        [*np.asarray(velocity_xy_global).flat[:2], 0.0], dtype=np.float64)
    velo = velo @ np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T
    return velo[:2].astype(np.float32)


def _add_annotation_velocity_prev_next(
    total_annotations,
    instance_token_mappings,
    lidar_ego_global_infos,
    v2x_side='vehicle-side',
    inf_labels_in_world_frame=False,
    max_workers=None
):
    """
    velocity 在 current frame lidar 下计算: (center_1 变换到 frame0 lidar - center_0) / time_delta
    与 vis_gt 完全一致：timestamp 始终 /1e6 转秒，变换公式一致
    inf_labels_in_world_frame: 若为 True 且 v2x_side==infrastructure-side，则 3d_location 视为 world，
        先做 global->lidar 转换再计算 velocity（用于 SPD 路端标注为 world 的修正）
    """
    if max_workers is None:
        max_workers = THREAD_POOL_CPU_WORKERS
    import json
    debug_file = open("velocity_abnormal_samples.json", "w")

    def process_instance(instance_token, samples):
        updated_samples = []
        num_samples = len(samples)
        for ii in range(num_samples):
            curr_sample = samples[ii]
            prev_anno_token = samples[ii -
                                      1]['annotation']['token'] if ii > 0 else ''
            next_anno_token = samples[ii +
                                      1]['annotation']['token'] if ii < num_samples - 1 else ''

            if ii == len(samples) - 1:
                gt_velocity = np.array([0, 0])
            else:
                loc_ii_0 = curr_sample['annotation']['3d_location']
                loc_ii_1 = samples[ii + 1]['annotation']['3d_location']
                sample_token_0 = curr_sample['sample_token']
                sample_token_1 = samples[ii + 1]['sample_token']

                center_0 = np.array(
                    [loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
                center_1 = np.array(
                    [loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])

                # Inf 路端：若 SPD 标注 3d_location 为 world 而非 lidar，先转换到 lidar
                if v2x_side == 'infrastructure-side' and inf_labels_in_world_frame:
                    center_0 = global_to_lidar(center_0, lidar_ego_global_infos[sample_token_0])
                    center_1 = global_to_lidar(center_1, lidar_ego_global_infos[sample_token_1])
                    # 此后 center_0/1 为 lidar 坐标，沿用下方标准变换
                # else: 标注已在 lidar，直接使用

                # # center_1: lidar1 -> ego1 -> global -> ego0 -> lidar0 (vis_gt 逻辑)
                l2e_1 = Quaternion(
                    lidar_ego_global_infos[sample_token_1]['lidar2ego_rotation']).rotation_matrix
                e2g_1 = Quaternion(
                    lidar_ego_global_infos[sample_token_1]['ego2global_rotation']).rotation_matrix
                center_1 = np.dot(l2e_1, center_1) + np.array(
                    lidar_ego_global_infos[sample_token_1]['lidar2ego_translation'])
                center_1 = np.dot(e2g_1, center_1) + np.array(
                    lidar_ego_global_infos[sample_token_1]['ego2global_translation'])

                e2g_0 = Quaternion(
                    lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix
                l2e_0 = Quaternion(
                    lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix

                global2ego_r0 = np.linalg.inv(e2g_0)
                global2ego_t0 = (-np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation']).reshape(
                    1, 3) @ global2ego_r0.T).reshape(3)

                ego2lidar_r0 = np.linalg.inv(l2e_0)
                ego2lidar_t0 = (-np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation']).reshape(
                    1, 3) @ ego2lidar_r0.T).reshape(3)

                center_1 = np.dot(global2ego_r0, center_1) + global2ego_t0
                center_1 = np.dot(ego2lidar_r0, center_1) + ego2lidar_t0

                # center_0 conver to global coordinate
                center_0 = np.dot(l2e_0, center_0) + np.array(
                    lidar_ego_global_infos[sample_token_0]['lidar2ego_translation'])

                center_0 = np.dot(e2g_0, center_0) + np.array(
                    lidar_ego_global_infos[sample_token_0]['ego2global_translation'])

                center_0 = np.dot(global2ego_r0, center_0) + global2ego_t0
                center_0 = np.dot(ego2lidar_r0, center_0) + ego2lidar_t0

                # 与 vis_gt 一致：timestamp 始终 /1e6 转为秒
                timestamp_ii_0 = float(curr_sample['timestamp']) / 1e6
                timestamp_ii_1 = float(samples[ii + 1]['timestamp']) / 1e6
                time_delta = timestamp_ii_1 - timestamp_ii_0
                if time_delta > 1e-6:
                    gt_velocity = (center_1 - center_0)[:2] / time_delta
                else:
                    gt_velocity = np.zeros(2)

            curr_sample['annotation']['gt_velocity'] = gt_velocity.tolist()
            curr_sample['annotation']['prev'] = prev_anno_token
            curr_sample['annotation']['next'] = next_anno_token
            updated_samples.append(curr_sample)

        return (instance_token, updated_samples)

    # 顺序执行，与 vis_gt 一致，确保结果确定性
    for instance_token, samples in tqdm(instance_token_mappings.items(), desc="velocity_prev_next"):
        _, updated_samples = process_instance(instance_token, samples)
        instance_token_mappings[instance_token] = updated_samples
        for sample_node in updated_samples:
            token = sample_node['sample_token']
            total_annotations[token][sample_node['annotation']
                                     ['token']] = sample_node['annotation']

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

    # 修复点：按时间戳排序，确保时间序列正确
    for i_token in instance_token_mappings:
        instance_token_mappings[i_token].sort(key=lambda x: x['frame_idx'])

    return instance_token_mappings

# ----------SPD datasets Process-------------------

# vis_gt 约定：timestamp 统一为微秒（/1e6 得秒）。原始可能是秒或微秒，此处规范化


def _timestamp_to_us(ts):
    ts = float(ts)
    return int(ts * 1e6) if ts < 1e12 else int(ts)


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
                'timestamp': _timestamp_to_us(sample_mappings[fid]['pointcloud_timestamp']),
                'image_timestamp': _timestamp_to_us(sample_mappings[fid]['image_timestamp']),
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
        scene_dict.setdefault(
            coop_data['vehicle_sequence'], []).append(veh_fid)

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
                'timestamp': _timestamp_to_us(veh_info['pointcloud_timestamp']),
                'image_timestamp': _timestamp_to_us(veh_info['image_timestamp']),
                'scene_token': veh_info['sequence_id'],
                'location': veh_info['intersection_loc'],
                'frame_idx': idx,
                'prev': frame_ids[idx-1] if idx > 0 else '',
                'next': frame_ids[idx+1] if idx < len(frame_ids)-1 else '',
                'token_inf': inf_fid,
                'timestamp_inf': _timestamp_to_us(inf_info['pointcloud_timestamp']),
                'image_timestamp_inf': _timestamp_to_us(inf_info['image_timestamp']),
                'system_error_offset': coop_info['system_error_offset']
            }
            sample_infos.append(info)

    sample_info_mappings = {info['token']: info for info in sample_infos}
    return sample_infos, sample_info_mappings


def _get_total_annotations(root_path, data_infos, sample_info_mappings, max_workers=None):
    """
    Load all annotation JSONs for vehicle-side or infrastructure-side dataset.

    Args:
        root_path (str): 数据根路径
        data_infos (list): 已加载的 data_info 列表
        sample_info_mappings (dict): sample_token -> sample_info
        max_workers (int): 多线程读取 JSON 数量（I/O bound，默认 THREAD_POOL_IO_WORKERS）

    Returns:
        dict: {frame_id: {anno_token: anno_dict}}
    """
    if max_workers is None:
        max_workers = THREAD_POOL_IO_WORKERS

    def load_single_annotation(data_info):
        sample_token = data_info['frame_id']
        scene_token = sample_info_mappings[sample_token]['scene_token']
        annotation_path = osp.join(
            root_path, data_info['label_lidar_std_path'])
        annos = load_json(annotation_path)
        ann_dict = {}
        for anno in annos:
            track_id = anno.get('track_id', -1)
            anno_token = anno.get('token', gen_token(track_id, scene_token))
            anno["instance_token"] = gen_token(track_id, scene_token)
            anno['type'] = class_names_nuscenes_mappings.get(
                anno.get('type', 'unknown'), 'unknown')
            ann_dict[anno_token] = anno
        return sample_token, ann_dict

    # 保持 data_infos 顺序，与 vis_gt 一致（instance_token_mappings 的 frame 顺序依赖此）
    total_annotations = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_annotation, di)
                   for di in data_infos]
        for f in tqdm(futures, desc="Loading annotations"):
            sample_token, ann_dict = f.result()
            total_annotations[sample_token] = ann_dict

    return total_annotations


# Worker state for ProcessPoolExecutor (set by initializer, read by worker)
_CREATE_SPD_WORKER_STATE = None


def _init_create_spd_worker(worker_state):
    """Set shared state for create_spd_infos workers (called once per process)."""
    global _CREATE_SPD_WORKER_STATE
    _CREATE_SPD_WORKER_STATE = worker_state


def _process_single_frame_worker(data_info):
    """Top-level worker for ProcessPoolExecutor. Reads state from _CREATE_SPD_WORKER_STATE."""
    s = _CREATE_SPD_WORKER_STATE
    if s is None:
        raise RuntimeError("Worker state not initialized")
    root_path = s['root_path']
    v2x_side = s['v2x_side']
    sample_info_mappings = s['sample_info_mappings']
    lidar_ego_global_infos = s['lidar_ego_global_infos']
    data_infos_mapping = s['data_infos_mapping']
    total_annotations = s['total_annotations']
    instance_token_mappings = s['instance_token_mappings']
    visibility_mappings = s['visibility_mappings']
    max_sweeps = s['max_sweeps']
    forecasting = s['forecasting']
    forecasting_length = s['forecasting_length']
    inf_labels_in_world_frame = s.get('inf_labels_in_world_frame', False)

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

    info['lidar_path'] = data_info['pointcloud_path']
    info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
    info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
    info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
    info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

    # vis_gt 一致：保持 sample_info['timestamp'] 原样（sweep 内部会按需转换）
    info['timestamp'] = sample_info['timestamp']

    camera_type = 'VEHICLE_CAM_FRONT'
    info['cams'] = {camera_type: {}}
    info['cams'][camera_type]['data_path'] = data_info['image_path']

    key_calib = 'calib_virtuallidar_to_camera_path' if v2x_side == 'infrastructure-side' else 'calib_lidar_to_camera_path'
    calib_lidar2cam_path = osp.join(root_path, data_info[key_calib])
    calib_lidar2cam = load_json(calib_lidar2cam_path)

    info['cams'][camera_type]['lidar2cam_rotation'] = np.array(
        calib_lidar2cam['rotation'])
    info['cams'][camera_type]['lidar2cam_translation'] = np.array(
        calib_lidar2cam['translation'])
    cam2lidar_r = np.linalg.inv(calib_lidar2cam['rotation'])
    cam2lidar_t = - \
        np.array(calib_lidar2cam['translation']).reshape(1, 3) @ cam2lidar_r.T
    info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
    info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(
        3)
    cam2ego_r, cam2ego_t = mul_matrix(
        cam2lidar_r, cam2lidar_t,
        Quaternion(info['lidar2ego_rotation']).rotation_matrix,
        np.array(info['lidar2ego_translation']))
    info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
    info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)

    calib_cam_intrinsic_path = osp.join(
        root_path, data_info['calib_camera_intrinsic_path'])
    info['cams'][camera_type]['cam_intrinsic'] = get_cam_intr(
        calib_cam_intrinsic_path)
    # Add camera timestamp (matching nuScenes format, in microseconds)
    # SPD timestamp is already in microseconds, no conversion needed
    info['cams'][camera_type]['timestamp'] = int(float(data_info.get(
        'image_timestamp', data_info.get('pointcloud_timestamp', 0))))

    info['sweeps'] = generate_sweeps(sample_token, sample_info_mappings, data_infos_mapping,
                                     lidar_ego_global_infos, max_sweeps=max_sweeps)
    info['can_bus'] = process_can_bus(
        info['ego2global_translation'], info['ego2global_rotation'])

    # annotations = total_annotations[sample_token]
    annotations = total_annotations.get(sample_token, {})
    boxes = []
    # 按 anno_token 排序，与 vis_gt 顺序一致，保证 pkl 输出顺序可对比
    for anno_token in sorted(annotations.keys()):
        annotation = annotations[anno_token]
        box3d = Box3D()
        raw_center = np.array([annotation['3d_location']['x'],
                               annotation['3d_location']['y'], annotation['3d_location']['z']])
        # Inf 路端：若 SPD 标注 3d_location 为 world，转换为 lidar（含 rotation）
        if v2x_side == 'infrastructure-side' and inf_labels_in_world_frame:
            raw_center = global_to_lidar(raw_center, lidar_ego_global_infos[sample_token])
            rot = annotation['rotation']
            yaw_rad = float(rot) if np.isscalar(rot) else float(rot[0])
            box3d.orientation_yaw_pitch_roll = global_yaw_to_lidar(
                yaw_rad, lidar_ego_global_infos[sample_token])
        else:
            box3d.orientation_yaw_pitch_roll = annotation['rotation']
        box3d.center = raw_center.tolist()
        # SPD 3d_dimensions: w=width(lateral), l=length(longitudinal)，与 SECOND 一致，无需互换
        # 与 vis_gt、spd_prediction_tools、spd_to_nuscenes 保持一致
        box3d.wlh = [annotation['3d_dimensions']['w'],
                     annotation['3d_dimensions']['l'], annotation['3d_dimensions']['h']]
        box3d.name = annotation['type']
        box3d.token = annotation['token']
        box3d.instance_token = annotation['instance_token']
        box3d.track_id = int(annotation['track_id'])
        box3d.timestamp = float(sample_info['timestamp'])  # vis_gt 一致
        box3d.visibility = visibility_mappings[annotation['occluded_state']]
        box3d.gt_velocity = annotation['gt_velocity']
        box3d.prev = annotation['prev']
        box3d.next = annotation['next']
        boxes.append(box3d)

    # gt_boxes: 3d_location is already in lidar frame, use directly
    if len(boxes) > 0:
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll
                        for b in boxes]).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
    else:
        gt_boxes = np.zeros((0, 7))
    info['gt_boxes'] = gt_boxes
    info['gt_names'] = np.array([b.name for b in boxes])
    info['gt_ins_tokens'] = np.array([b.instance_token for b in boxes])
    info['gt_inds'] = np.array([b.track_id for b in boxes])
    info['anno_tokens'] = np.array([b.token for b in boxes])
    info['timestamps'] = np.array([b.timestamp for b in boxes])
    info['visibility_tokens'] = np.array([b.visibility for b in boxes])
    # 套用 vis_gt：velocity 已在 current frame lidar 下计算，直接使用
    info['gt_velocity'] = np.array(
        [b.gt_velocity for b in boxes], dtype=np.float32)
    info['prev_anno_tokens'] = np.array([b.prev for b in boxes])
    info['next_anno_tokens'] = np.array([b.next for b in boxes])

    # num_pts 过滤：按 box 内点云数量设置 valid_flag（dataset 会用 mask 过滤）
    lidar_full_path = osp.join(root_path, data_info['pointcloud_path'])
    num_lidar_pts_arr = compute_num_lidar_pts_per_box(
        lidar_full_path, gt_boxes, load_dim=5)
    info['num_lidar_pts'] = num_lidar_pts_arr
    info['valid_flag'] = num_lidar_pts_arr >= 1

    attributes, attribute_tokens = [], []
    for name, vel, vis, valid in zip(info['gt_names'], info['gt_velocity'], info['visibility_tokens'], info['valid_flag']):
        attr = infer_nuscenes_attribute(name, vel, vis, valid)
        if attr is None:
            attributes.append('None')
            attribute_tokens.append(-1)
        else:
            attributes.append(attr)
            attribute_tokens.append(ATTRIBUTE_MAPPING[attr])
    info['gt_attributes'] = np.array(attributes, dtype=object)
    info['gt_attribute_tokens'] = np.array(attribute_tokens, dtype=np.int64)

    if forecasting:
        fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(
            instance_token_mappings, lidar_ego_global_infos, annotations, forecasting_length)
        info['forecasting_locs'] = np.array(
            [np.array([b.center for b in fb]).reshape(-1, 3) for fb in fboxes])
        info['forecasting_tokens'] = np.array(
            [np.array([b.token for b in fb]) for fb in fboxes])
        info['forecasting_masks'] = np.array(fmasks)
        info['forecasting_types'] = np.array(ftypes)

    return info


# Worker state for create_spd_infos_coop (ProcessPoolExecutor)
_CREATE_SPD_COOP_WORKER_STATE = None


def _init_create_spd_coop_worker(worker_state):
    """Set shared state for create_spd_infos_coop workers (called once per process)."""
    global _CREATE_SPD_COOP_WORKER_STATE
    _CREATE_SPD_COOP_WORKER_STATE = worker_state


def _process_single_frame_coop_worker(coop_data_info):
    """Top-level worker for create_spd_infos_coop ProcessPoolExecutor."""
    s = _CREATE_SPD_COOP_WORKER_STATE
    if s is None:
        raise RuntimeError("Coop worker state not initialized")
    root_path = s['root_path']
    sample_info_mappings = s['sample_info_mappings']
    lidar_ego_global_infos = s['lidar_ego_global_infos']
    veh_map = s['veh_map']
    inf_map = s['inf_map']
    sweep_data_mapping = s['sweep_data_mapping']
    total_annotations = s['total_annotations']
    instance_token_mappings = s['instance_token_mappings']
    visibility_mappings = s['visibility_mappings']
    max_sweeps = s['max_sweeps']
    forecasting = s['forecasting']
    forecasting_length = s['forecasting_length']

    veh_frame_id = coop_data_info['vehicle_frame']
    inf_frame_id = coop_data_info['infrastructure_frame']
    sample_token = veh_frame_id
    sample_info = sample_info_mappings[sample_token]

    info = dict(
        token=sample_info['token'],
        frame_idx=sample_info['frame_idx'],
        scene_token=sample_info['scene_token'],
        location=sample_info['location'],
        timestamp=sample_info['timestamp'],
        prev=sample_info['prev'],
        next=sample_info['next'],
        token_inf=sample_info['token_inf'],
        timestamp_inf=sample_info['timestamp_inf'],
        system_error_offset=sample_info['system_error_offset']
    )

    info['timestamp'] = sample_info['timestamp']  # vis_gt 一致
    info['timestamp_inf'] = sample_info['timestamp_inf']

    veh_data_info = veh_map[veh_frame_id]
    inf_data_info = inf_map[inf_frame_id]

    info['lidar_path'] = veh_data_info['pointcloud_path']
    info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
    info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
    info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
    info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

    # Infrastructure (ego = virtuallidar)
    info['lidar2ego_rotation_inf'] = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    info['lidar2ego_translation_inf'] = np.zeros(3, dtype=np.float32)
    calib_vl2g = load_json(osp.join(
        root_path, 'infrastructure-side',
        inf_data_info['calib_virtuallidar_to_world_path']))
    virtuallidar2world_r = np.array(calib_vl2g['rotation']).reshape(3, 3)
    virtuallidar2world_r = orthonormalize_rotation(virtuallidar2world_r)
    try:
        q_inf = Quaternion(matrix=virtuallidar2world_r)
    except ValueError:
        virtuallidar2world_r = iterative_closest_point(
            virtuallidar2world_r.astype(np.float64))
        q_inf = Quaternion(matrix=virtuallidar2world_r)
    info['ego2global_rotation_inf'] = np.array(
        [q_inf.w, q_inf.x, q_inf.y, q_inf.z], dtype=np.float32)
    info['ego2global_translation_inf'] = np.array(
        calib_vl2g['translation'], dtype=np.float32).reshape(3)

    # VehLidar2InfLidar (coop coordinate transform)
    veh_l2e_r = np.array(Quaternion(
        info['lidar2ego_rotation']).rotation_matrix)
    veh_l2e_t = np.array(info['lidar2ego_translation']).reshape(3)
    veh_e2g_r = np.array(Quaternion(
        info['ego2global_rotation']).rotation_matrix)
    veh_e2g_t = np.array(info['ego2global_translation']).reshape(3)
    inf_e2g_r = np.array(calib_vl2g['rotation']).reshape(3, 3)
    inf_e2g_t = np.array(calib_vl2g['translation']).reshape(3)
    inf_l2e_r = np.eye(3)
    inf_l2e_t = np.zeros(3)
    err_offset = np.array([
        sample_info['system_error_offset']['delta_x'],
        sample_info['system_error_offset']['delta_y'], 0])
    r = ((veh_l2e_r.T @ veh_e2g_r.T) @
         (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
    t = (-err_offset @ veh_l2e_r.T @ veh_e2g_r.T + veh_l2e_t @ veh_e2g_r.T +
         veh_e2g_t) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
    t -= inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) + \
        inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T)
    info['VehLidar2InfLidar_rotation'] = r
    info['VehLidar2InfLidar_translation'] = t

    # Camera
    camera_type = 'VEHICLE_CAM_FRONT'
    info['cams'] = {camera_type: {}}
    info['cams'][camera_type]['data_path'] = veh_data_info['image_path']
    calib_l2c = load_json(osp.join(
        root_path, 'vehicle-side',
        veh_data_info['calib_lidar_to_camera_path']))
    info['cams'][camera_type]['lidar2cam_rotation'] = np.array(
        calib_l2c['rotation'])
    info['cams'][camera_type]['lidar2cam_translation'] = np.array(
        calib_l2c['translation'])
    cam2lidar_r = np.linalg.inv(calib_l2c['rotation'])
    cam2lidar_t = - \
        np.array(calib_l2c['translation']).reshape(1, 3) @ cam2lidar_r.T
    info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
    info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(
        3)
    cam2ego_r, cam2ego_t = mul_matrix(
        cam2lidar_r, cam2lidar_t,
        Quaternion(info['lidar2ego_rotation']).rotation_matrix,
        np.array(info['lidar2ego_translation']))
    info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
    info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)
    info['cams'][camera_type]['cam_intrinsic'] = get_cam_intr(osp.join(
        root_path, 'vehicle-side',
        veh_data_info['calib_camera_intrinsic_path']))
    # Add camera timestamp (matching nuScenes format, in microseconds)
    # SPD timestamp is already in microseconds, no conversion needed
    info['cams'][camera_type]['timestamp'] = int(float(veh_data_info.get(
        'image_timestamp', veh_data_info.get('pointcloud_timestamp', 0))))

    info['sweeps'] = generate_sweeps(
        sample_token, sample_info_mappings, sweep_data_mapping,
        lidar_ego_global_infos, max_sweeps=max_sweeps)
    info['can_bus'] = process_can_bus(
        info['ego2global_translation'], info['ego2global_rotation'])

    # SPD label_lidar_std_path: 3d_location is in LiDAR frame
    annotations = total_annotations[sample_token]
    boxes = []
    for anno_token in sorted(annotations.keys()):
        annotation = annotations[anno_token]
        box3d = Box3D()
        box3d.center = [
            annotation['3d_location']['x'],
            annotation['3d_location']['y'],
            annotation['3d_location']['z']
        ]
        # SPD 3d_dimensions: w=width(lateral), l=length(longitudinal)，与 SECOND 一致，无需互换
        box3d.wlh = [
            annotation['3d_dimensions']['w'],
            annotation['3d_dimensions']['l'],
            annotation['3d_dimensions']['h']
        ]
        box3d.orientation_yaw_pitch_roll = annotation['rotation']
        box3d.name = annotation['type']
        box3d.token = annotation['token']
        box3d.instance_token = annotation['instance_token']
        box3d.track_id = int(annotation['track_id'])
        box3d.timestamp = float(sample_info['timestamp'])  # vis_gt 一致
        box3d.visibility = visibility_mappings[annotation['occluded_state']]
        box3d.gt_velocity = annotation['gt_velocity']  # in global frame
        box3d.prev = annotation['prev']
        box3d.next = annotation['next']
        boxes.append(box3d)

    # gt_boxes: 3d_location is already in lidar frame, use directly
    if len(boxes) > 0:
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll
                        for b in boxes]).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        # num_pts 过滤：按 box 内点云数量设置 valid_flag
        lidar_full_path = osp.join(
            root_path, 'vehicle-side', veh_data_info['pointcloud_path'])
        num_lidar_pts = compute_num_lidar_pts_per_box(
            lidar_full_path, gt_boxes, load_dim=5)
        valid_flag = num_lidar_pts >= 1
    else:
        gt_boxes = np.zeros((0, 7))
        valid_flag = np.array([], dtype=bool)
        num_lidar_pts = np.array([], dtype=np.int64)

    names = np.array([b.name for b in boxes])
    visibility_tokens = np.array([b.visibility for b in boxes])
    # 套用 vis_gt：velocity 已在 lidar 下，直接使用
    gt_velocity = np.array([b.gt_velocity for b in boxes], dtype=np.float32)

    attributes, attribute_tokens = [], []
    for name, vel, vis, valid in zip(names, gt_velocity, visibility_tokens, valid_flag):
        attr = infer_nuscenes_attribute(name, vel, vis, valid)
        if attr is None:
            attributes.append('None')
            attribute_tokens.append(-1)
        else:
            attributes.append(attr)
            attribute_tokens.append(ATTRIBUTE_MAPPING[attr])

    info.update(dict(
        gt_boxes=gt_boxes,
        gt_names=names,
        gt_velocity=gt_velocity,
        visibility_tokens=visibility_tokens,
        valid_flag=valid_flag,
        num_lidar_pts=num_lidar_pts,
        gt_attributes=np.array(attributes, dtype=object),
        gt_attribute_tokens=np.array(attribute_tokens, dtype=np.int64),
        gt_ins_tokens=np.array([b.instance_token for b in boxes]),
        gt_inds=np.array([b.track_id for b in boxes]),
        anno_tokens=np.array([b.token for b in boxes]),
        timestamps=np.array([b.timestamp for b in boxes]),
        prev_anno_tokens=np.array([b.prev for b in boxes]),
        next_anno_tokens=np.array([b.next for b in boxes]),
    ))

    if forecasting:
        fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(
            instance_token_mappings,
            lidar_ego_global_infos,
            annotations,
            forecasting_length)
        locs = [np.array([b.center for b in fb]).reshape(-1, 3)
                for fb in fboxes]
        tokens = [np.array([b.token for b in fb]) for fb in fboxes]
        info['forecasting_locs'] = np.array(locs)
        info['forecasting_tokens'] = np.array(tokens)
        info['forecasting_masks'] = np.array(fmasks)
        info['forecasting_types'] = np.array(ftypes)

    return info


def _get_total_annotations_coop(root_path, data_infos, sample_info_mappings, max_workers=None):
    if max_workers is None:
        max_workers = THREAD_POOL_IO_WORKERS

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
            anno['type'] = class_names_nuscenes_mappings.get(
                anno.get('type', 'unknown'), 'unknown')
            ann_dict[anno_token] = anno
        return veh_fid, ann_dict

    total_annotations = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_annotation, d)
                   for d in data_infos]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Loading coop annotations"):
            veh_fid, ann_dict = f.result()
            total_annotations[veh_fid] = ann_dict

    return total_annotations


def create_spd_infos(root_path,
                     out_path,
                     v2x_side,
                     split_path,
                     info_prefix,
                     version='v1.0-trainval',
                     max_sweeps=10,
                     split_part='train',
                     flag_save=True,
                     forecasting=False,
                     forecasting_length=13,
                     max_workers=None,
                     inf_labels_in_world_frame=False):
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
        data_infos = [
            item for item in data_infos if item['frame_id'] not in to_remove_list_veh]

    # Generate sample mappings and annotations
    sample_infos, sample_info_mappings = _generate_sample_infos(data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations(
        root_path, data_infos, sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(
        total_annotations, sample_info_mappings)

    # Get lidar -> ego -> global info
    lidar_ego_global_infos = get_lidar_ego_global_infos(
        root_path, data_infos, v2x_side)

    # Interpolate unvisible objects
    total_annotations = _generate_unvisible_annotations(v2x_side, sample_info_mappings, secene_frame_mappings,
                                                        instance_token_mappings, total_annotations, lidar_ego_global_infos)

    # Update instance token mappings
    instance_token_mappings = _get_instance_token_mappings(
        total_annotations, sample_info_mappings)

    # Add velocity and prev/next
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(
        total_annotations, instance_token_mappings, lidar_ego_global_infos,
        v2x_side=v2x_side, inf_labels_in_world_frame=inf_labels_in_world_frame)

    # Precompute once
    data_infos_mapping = _build_sweep_data_mapping(
        data_infos, lidar_ego_global_infos)

    # ---------------------------------------------------------------------
    # ProcessPoolExecutor: true multi-process (bypass GIL), state via initializer
    # ---------------------------------------------------------------------
    if max_workers is None:
        max_workers = CREATE_SPD_INFOS_WORKERS

    worker_state = {
        'root_path': root_path,
        'v2x_side': v2x_side,
        'sample_info_mappings': sample_info_mappings,
        'lidar_ego_global_infos': lidar_ego_global_infos,
        'data_infos_mapping': data_infos_mapping,
        'total_annotations': total_annotations,
        'instance_token_mappings': instance_token_mappings,
        'visibility_mappings': visibility_mappings,
        'max_sweeps': max_sweeps,
        'forecasting': forecasting,
        'forecasting_length': forecasting_length,
        'inf_labels_in_world_frame': inf_labels_in_world_frame,
    }

    spd_infos = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_create_spd_worker, initargs=(worker_state,)) as executor:
        futures = [executor.submit(
            _process_single_frame_worker, data_info) for data_info in data_infos]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="create_spd_infos"):
            spd_infos.append(fut.result())

    # Split train / val
    train_spd_infos = [
        info for info in spd_infos if sample_info_mappings[info['token']]['scene_token'] in train_scenes]
    val_spd_infos = [
        info for info in spd_infos if sample_info_mappings[info['token']]['scene_token'] in val_scenes]

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
                          max_workers=None):

    out_path = osp.join(out_path, v2x_side)

    # ===============================
    # Load data
    # ===============================

    coop_data_infos = load_json(
        osp.join(root_path, 'cooperative/data_info.json'))
    coop_data_infos = [
        item for item in coop_data_infos
        if item['vehicle_frame'] not in to_remove_list_coop
    ]

    veh_data_infos = load_json(
        osp.join(root_path, 'vehicle-side/data_info.json'))

    inf_data_infos = load_json(
        osp.join(root_path, 'infrastructure-side/data_info.json'))

    split_data = load_json(split_path)
    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    # ===============================
    # Annotation & mapping
    # ===============================

    sample_infos, sample_info_mappings = _generate_sample_infos_coop(
        coop_data_infos, veh_data_infos, inf_data_infos)

    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)

    total_annotations = _get_total_annotations_coop(
        root_path, coop_data_infos, sample_info_mappings)

    instance_token_mappings = _get_instance_token_mappings(
        total_annotations, sample_info_mappings)

    lidar_ego_global_infos = get_lidar_ego_global_infos(
        osp.join(root_path, 'vehicle-side'),
        veh_data_infos,
        v2x_side)

    total_annotations = _generate_unvisible_annotations(
        "cooperative",
        sample_info_mappings,
        secene_frame_mappings,
        instance_token_mappings,
        total_annotations,
        lidar_ego_global_infos)

    instance_token_mappings = _get_instance_token_mappings(
        total_annotations, sample_info_mappings)

    total_annotations, instance_token_mappings = \
        _add_annotation_velocity_prev_next(
            total_annotations,
            instance_token_mappings,
            lidar_ego_global_infos)

    # 预索引
    veh_map = {d['frame_id']: d for d in veh_data_infos}
    inf_map = {d['frame_id']: d for d in inf_data_infos}
    sweep_data_mapping = _build_sweep_data_mapping(
        veh_data_infos, lidar_ego_global_infos)

    # ===============================
    # ProcessPoolExecutor (与 single 一致，避免 pickle 闭包)
    # ===============================
    if max_workers is None:
        max_workers = CREATE_SPD_INFOS_WORKERS

    coop_worker_state = {
        'root_path': root_path,
        'sample_info_mappings': sample_info_mappings,
        'lidar_ego_global_infos': lidar_ego_global_infos,
        'veh_map': veh_map,
        'inf_map': inf_map,
        'sweep_data_mapping': sweep_data_mapping,
        'total_annotations': total_annotations,
        'instance_token_mappings': instance_token_mappings,
        'visibility_mappings': visibility_mappings,
        'max_sweeps': max_sweeps,
        'forecasting': forecasting,
        'forecasting_length': forecasting_length,
    }

    spd_infos = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_create_spd_coop_worker, initargs=(coop_worker_state,)) as executor:
        futures = [executor.submit(
            _process_single_frame_coop_worker, d) for d in coop_data_infos]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="create_spd_infos_coop"):
            spd_infos.append(fut.result())

    train_spd_infos = [
        info for info in spd_infos
        if sample_info_mappings[info['token']]['scene_token'] in train_scenes
    ]

    val_spd_infos = [
        info for info in spd_infos
        if sample_info_mappings[info['token']]['scene_token'] in val_scenes
    ]

    if flag_save:
        metadata = dict(version=version)

        mmcv.dump(
            dict(infos=train_spd_infos, metadata=metadata),
            osp.join(out_path, f'{info_prefix}_infos_temporal_train.pkl')
        )

        mmcv.dump(
            dict(infos=val_spd_infos, metadata=metadata),
            osp.join(out_path, f'{info_prefix}_infos_temporal_val.pkl')
        )

    return total_annotations, sample_info_mappings, spd_infos


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str,
                        default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--save-root', type=str,
                        default="./data/infos/V2X-Seq-SPD-Example")
    parser.add_argument('--split-file', type=str,
                        default="./data/split_datas/cooperative-split-data-spd.json")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="spd")
    parser.add_argument('--forecasting', action='store_true',
                        default=False, help='prepare trajectory forecasting data')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Process workers for create_spd_infos (default: CREATE_SPD_INFOS_WORKERS)')
    parser.add_argument('--inf-labels-in-world-frame', action='store_true',
                        help='Infrastructure: 3d_location in label is world frame, convert to lidar (fix GT/pred offset)')

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

    # spd_to_uniad: 只生成 uniad 格式，不生成 pkl（pkl 由 spd_to_nuscenes 生成）
    if v2x_side == 'cooperative':
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos_coop(
            data_root,
            save_root,
            v2x_side,
            split_path,
            can_bus_root_path,
            info_prefix,
            version=args.version,
            max_sweeps=10,
            forecasting=forecasting,
            max_workers=args.max_workers,
            flag_save=False)
    else:
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos(
            data_root,
            save_root,
            v2x_side,
            split_path,
            info_prefix,
            version=args.version,
            max_sweeps=10,
            forecasting=forecasting,
            max_workers=args.max_workers,
            flag_save=False,
            inf_labels_in_world_frame=args.inf_labels_in_world_frame)
