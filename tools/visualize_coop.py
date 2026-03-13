import struct
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import os
import cv2
import pickle
import matplotlib
import numpy as np
import os.path as osp
# import open3d as o3d
import matplotlib.pyplot as plt
import struct
from matplotlib.patches import Polygon
matplotlib.use('Agg')
# ===============================
# 数学工具
# ===============================


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


def limit_period(val):
    return (val + np.pi) % (2 * np.pi) - np.pi


def _load_original_lidar_labels(data_root, frame_id):
    """
    加载原始 lidar 标注 JSON（vehicle-side/label/lidar/{frame_id}.json）。
    返回 annotations 列表，失败返回 None。
    """
    for prefix in ['vehicle-side', '']:
        path = osp.join(data_root, prefix, 'label',
                        'lidar', f'{frame_id}.json')
        if osp.isfile(path):
            with open(path, 'r') as f:
                return json.load(f)
    return None


def _lidar_label_json_to_boxes(annotations):
    """
    将原始 label JSON 转为 (N, 7) box 格式：(x,y,z,w,l,h,yaw)，yaw 为 SECOND 约定 (-rot-pi/2)。
    """
    if not annotations:
        return np.zeros((0, 7), dtype=np.float32)
    boxes = []
    for ann in annotations:
        loc = ann['3d_location']
        dim = ann['3d_dimensions']
        rot = ann['rotation']  # 原始 yaw (rad)
        x, y, z = loc['x'], loc['y'], loc['z']
        w, l, h = dim['w'], dim['l'], dim['h']
        yaw = -rot - np.pi / 2  # SECOND
        boxes.append([x, y, z, w, l, h, yaw])
    return np.array(boxes, dtype=np.float32)


def _compute_num_pts_per_box(pts_xyz, boxes):
    """计算每个 box 内的点云数量。boxes: (N,7) (x,y,z,w,l,h,yaw) SECOND 格式。"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    w, l, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    yaw = boxes[:, 6]
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    dx = pts_xyz[:, 0] - cx[:, None]
    dy = pts_xyz[:, 1] - cy[:, None]
    dz = pts_xyz[:, 2] - cz[:, None]
    local_x = cos_yaw[:, None] * dx + sin_yaw[:, None] * dy
    local_y = -sin_yaw[:, None] * dx + cos_yaw[:, None] * dy
    in_x = np.abs(local_x) <= (l / 2)[:, None]
    in_y = np.abs(local_y) <= (w / 2)[:, None]
    in_z = np.abs(dz) <= (h / 2)[:, None]
    inside = in_x & in_y & in_z
    return inside.sum(axis=1).astype(np.int64)


def _points_lidar_to_img(pts_lidar, lidar2cam_r, lidar2cam_t, cam_intrinsic):
    """Project lidar points to image. pts_lidar (N,3), returns (N,2) uv, (N,) depth, (N,) valid."""
    pts_cam = (lidar2cam_r @ pts_lidar.T).T + \
        np.array(lidar2cam_t).reshape(1, 3)
    pts_2d = (cam_intrinsic @ pts_cam.T).T
    depth = pts_cam[:, 2]
    valid = depth > 0.1
    uv = np.zeros((len(pts_lidar), 2))
    uv[valid, 0] = pts_2d[valid, 0] / pts_2d[valid, 2]
    uv[valid, 1] = pts_2d[valid, 1] / pts_2d[valid, 2]
    return uv, depth, valid


def _box_center_in_cam_fov(box, lidar2cam_r, lidar2cam_t, cam_intrinsic, imsize):
    """Check if box center projects inside image FOV. box (7,), imsize (W,H)."""
    center = np.array([[box[0], box[1], box[2]]])
    uv, depth, valid = _points_lidar_to_img(
        center, lidar2cam_r, lidar2cam_t, cam_intrinsic)
    if not valid[0] or depth[0] <= 0:
        return False
    u, v = uv[0, 0], uv[0, 1]
    w, h = imsize[0], imsize[1]
    return 0 <= u < w and 0 <= v < h


def _get_camera_fov_bev_polygon(lidar2cam_r, lidar2cam_t, cam_intrinsic, imsize, depth=50.0):
    """Get camera FOV boundary polygon in BEV (lidar xy). Returns (5,2) or None."""
    w, h = imsize[0], imsize[1]
    # 图像四角 + 中心底边
    corners_2d = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, h]  # 中心底部
    ], dtype=np.float64)
    Kinv = np.linalg.inv(cam_intrinsic)
    cam2lidar_r = lidar2cam_r.T
    cam2lidar_t = -cam2lidar_r @ np.array(lidar2cam_t).reshape(3)
    pts_bev = []
    for i in range(5):
        uv = corners_2d[i]
        ray_cam = Kinv @ np.array([uv[0], uv[1], 1.0])
        if np.abs(ray_cam[2]) < 1e-6:
            continue
        # 取深度 depth 处的 3D 点 (相机系)
        p_cam = depth * ray_cam / ray_cam[2]
        p_lidar = cam2lidar_r @ p_cam + cam2lidar_t
        pts_bev.append(p_lidar[:2])
    if len(pts_bev) < 3:
        return None
    # 按角度排序成凸多边形（以原点为参考）
    pts = np.array(pts_bev)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(angles)
    return pts[order]


def compute_iou_bev(box1, box2):
    from shapely.geometry import Polygon as ShapelyPolygon

    def box_to_poly(box):
        # box: (x,y,z,w,l,h,yaw)，dx=length(前向)，dy=width(侧向)
        x, y, z, w, l, h, yaw = box[:7]
        dx, dy = l, w
        corners = np.array([
            [dx/2,  dy/2],
            [dx/2, -dy/2],
            [-dx/2, -dy/2],
            [-dx/2,  dy/2],
        ])
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        corners = corners @ R.T + np.array([x, y])
        return ShapelyPolygon(corners)

    poly1 = box_to_poly(box1)
    poly2 = box_to_poly(box2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0


def _lidar_bev_to_pixel(x, y, H, W, x_range=(-60, 60), y_range=(-60, 60)):
    """Map lidar (x,y) to image pixel (px, py). x=forward, y=left. Origin at center."""
    px = (x - x_range[0]) / (x_range[1] - x_range[0]) * (W - 1)
    py = (y_range[1] - y) / (y_range[1] - y_range[0]) * (H - 1)  # flip y
    return int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))


def _render_single_bev_frame_cv(args):
    """
    快速 OpenCV 渲染单帧 BEV，替代 matplotlib，约 10-50x 加速。
    Returns (idx, img) or (idx, None) on failure.
    """
    idx, info, data_root, frame_size, downsample, draw_text = args
    H, W = frame_size[1], frame_size[0]
    x_range, y_range = (-60, 60), (-60, 60)

    def box_corners_bev(box):
        # box: (x,y,z,w,l,h,yaw) SECOND 格式；BEV 局部 x=前向(length)，y=左(width)
        x, y, z, w, l, h, yaw = box[:7]
        dx, dy = l, w  # 前向用 length，侧向用 width
        corners = np.array([
            [dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2]
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = corners @ R.T + np.array([x, y])
        return corners

    try:
        pts = _load_points_xyz(osp.join(data_root, info['lidar_path']))
    except Exception:
        return (idx, None)

    # 下采样点云
    if downsample > 1:
        pts = pts[::downsample]

    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    # 绘制点云（投影到像素）
    px = ((pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (W - 1)).astype(np.int32)
    py = ((y_range[1] - pts[:, 1]) / (y_range[1] - y_range[0]) * (H - 1)).astype(np.int32)
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    img[py[valid], px[valid]] = 0

    boxes = info.get('gt_boxes', [])
    velocities = info.get('gt_velocity', [None] * len(boxes))
    track_ids = info.get('track_ids', [None] * len(boxes))
    classes = info.get('gt_classes', [None] * len(boxes))

    VEL_ABNORMAL_THRESH = 30.0  # m/s，超过此值视为异常速度

    for i, box in enumerate(boxes):
        corners = box_corners_bev(box)
        pts_px = np.array([_lidar_bev_to_pixel(c[0], c[1], H, W, x_range, y_range) for c in corners])
        pts_px = pts_px.reshape((-1, 1, 2)).astype(np.int32)
        vel_norm = np.linalg.norm(velocities[i]) if velocities[i] is not None else 0.0
        box_color = (0, 0, 255) if vel_norm > VEL_ABNORMAL_THRESH else (0, 255, 0)  # BGR: 红/绿
        cv2.polylines(img, [pts_px], True, box_color, 1)
        if velocities[i] is not None:
            vx, vy = velocities[i][:2]
            x2, y2 = box[0] + vx * 0.5, box[1] + vy * 0.5
            p1 = _lidar_bev_to_pixel(box[0], box[1], H, W, x_range, y_range)
            p2 = _lidar_bev_to_pixel(x2, y2, H, W, x_range, y_range)
            cv2.arrowedLine(img, p1, p2, (0, 0, 255), 1, tipLength=0.2)
        if draw_text:
            cx, cy = _lidar_bev_to_pixel(box[0], box[1], H, W, x_range, y_range)
            txt = ""
            if track_ids[i] is not None:
                txt += f"ID:{track_ids[i]} "
            if classes[i] is not None:
                txt += f"Cls:{classes[i]} "
            txt += f"Frm:{idx}"
            cv2.putText(img, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    return (idx, img)


def _render_single_bev_frame(args):
    """
    Worker for export_video: 使用 OpenCV 快速渲染 (fast=True 时) 或 matplotlib 渲染。
    Returns (idx, img) or (idx, None) on failure.
    """
    # 新接口支持 (idx, info, data_root, frame_size, downsample, draw_text)
    if len(args) >= 6:
        return _render_single_bev_frame_cv(args)
    # 兼容旧调用
    idx, info, data_root, frame_size = args[:4]
    return _render_single_bev_frame_cv((
        idx, info, data_root, frame_size,
        5,   # downsample
        False  # draw_text off for speed
    ))


# ===============================
# 主类
# ===============================

class SPDVisualizer:
    def __init__(self,
                 info_path,
                 data_root,
                 pred_path=None,
                 v2x_side='vehicle',
                 max_frames=None,
                 coop=False):

        self.data_root = data_root
        self.v2x_side = v2x_side
        self.coop = coop
        self.max_frames = max_frames
        with open(info_path, 'rb') as f:
            raw = pickle.load(f)

        # 自动兼容 dict / list 两种格式
        if isinstance(raw, dict):
            if 'infos' in raw:
                self.infos = raw['infos']
            else:
                self.infos = list(raw.values())
        else:
            self.infos = raw

        self.preds = None
        if pred_path is not None:
            with open(pred_path, 'rb') as f:
                self.preds = pickle.load(f)

        self._veh_data_infos = None  # cache for coop cam loading

    def _load_cams_for_coop_frame(self, frame_token):
        """当 info 无 cams 时，尝试从 vehicle-side 加载相机标定（用于 coop 数据）。"""
        if self._veh_data_infos is None:
            # 支持 data_root 为 project root 或 vehicle-side root
            for candidate in [osp.join(self.data_root, 'vehicle-side'), self.data_root]:
                data_info_path = osp.join(candidate, 'data_info.json')
                if osp.isfile(data_info_path):
                    with open(data_info_path, 'r') as f:
                        all_infos = json.load(f)
                    self._veh_data_infos = {
                        d['frame_id']: d for d in all_infos}
                    self._veh_root = candidate
                    break
            else:
                self._veh_data_infos = {}
                return None
        di = self._veh_data_infos.get(frame_token)
        if di is None:
            return None
        root = getattr(self, '_veh_root', self.data_root)
        try:
            calib = json.load(
                open(osp.join(root, di['calib_lidar_to_camera_path'])))
            intr_path = osp.join(root, di['calib_camera_intrinsic_path'])
            intr_data = json.load(open(intr_path))
            intr = np.array(intr_data.get('P', intr_data.get(
                'cam_K', np.eye(3)))).reshape(3, 4)[:3, :3]
            return {
                'lidar2cam_rotation': np.array(calib['rotation']),
                'lidar2cam_translation': np.array(calib['translation']),
                'cam_intrinsic': intr,
                'data_path': osp.join(root, di['image_path']),
                'data_path_is_absolute': True
            }
        except Exception:
            return None

    # ===============================
    # 读取点云
    # ===============================

    def load_lidar(self, path, load_dim=4):
        """
        支持 .bin / .pcd (ascii / binary / binary_compressed)
        """
        import os.path as osp
        full_path = osp.join(self.data_root, path)

        candidates = [full_path]
        if full_path.endswith('.bin'):
            candidates.append(full_path[:-4] + '.pcd')

        for p in candidates:
            if not osp.isfile(p):
                continue
            if p.endswith('.bin'):
                pts = np.fromfile(p, dtype=np.float32).reshape(-1, load_dim)
                return pts[:, :3]
            if p.endswith('.pcd'):
                pts = _load_points_xyz_from_pcd(p)
                return pts

        raise FileNotFoundError(f"Lidar file not found: {full_path}")

    # ===============================
    # box -> corners
    # ===============================

    def box_to_corners(self, box):
        # box: (x,y,z,w,l,h,yaw)，局部 x=length(前向)，y=width(侧向)，z=height
        x, y, z, w, l, h, yaw = box[:7]
        dx, dy, dz = l, w, h

        corners = np.array([
            [dx/2,  dy/2, -dz/2],
            [dx/2, -dy/2, -dz/2],
            [-dx/2, -dy/2, -dz/2],
            [-dx/2,  dy/2, -dz/2],
            [dx/2,  dy/2,  dz/2],
            [dx/2, -dy/2,  dz/2],
            [-dx/2, -dy/2,  dz/2],
            [-dx/2,  dy/2,  dz/2],
        ])

        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        corners = corners @ R.T + np.array([x, y, z])
        return corners

    # ===============================
    # BEV 可视化
    # ===============================

    def visualize_bev(self, pts, boxes, velocities=None, preds=None):

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3)

        for i, box in enumerate(boxes):
            corners = self.box_to_corners(box)[:4, :2]
            poly = Polygon(corners, fill=False)
            ax.add_patch(poly)

            if velocities is not None:
                vx, vy = velocities[i]
                ax.arrow(box[0], box[1], vx, vy, head_width=0.3)

        if preds is not None:
            for box in preds:
                corners = self.box_to_corners(box)[:4, :2]
                poly = Polygon(corners, fill=False, linestyle='--')
                ax.add_patch(poly)

        ax.set_aspect('equal')
        plt.title(f'BEV - {self.v2x_side}')
        plt.show()

    # ===============================
    # BEV + 相机 FOV 叠加（验证 GT 是否仅在图像区域内）
    # ===============================

    def visualize_bev_with_fov(self, pts, boxes, info, velocities=None, save_path=None):
        """
        BEV 可视化，叠加相机 FOV 区域。绿框=在图像 FOV 内，红框=在 FOV 外。
        用于验证点云 GT 是否仅针对图像区域标注。
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c='gray', alpha=0.5)

        # 尝试获取相机 FOV 并绘制
        cams = info.get('cams') or {}
        if not cams and hasattr(self, '_load_cams_for_coop_frame'):
            cam_data = self._load_cams_for_coop_frame(info.get('token', ''))
            if cam_data:
                cams = {'VEHICLE_CAM_FRONT': cam_data}
        cam_keys = [k for k in cams.keys()]
        fov_poly = None
        in_fov_mask = np.zeros(len(boxes), dtype=bool)
        imsize = (1920, 1080)

        if cam_keys:
            cam_name = cam_keys[0]
            cam = cams[cam_name]
            lidar2cam_r = np.array(cam.get('lidar2cam_rotation'))
            lidar2cam_t = np.array(cam.get('lidar2cam_translation')).reshape(3)
            intr = np.array(cam.get('cam_intrinsic', cam.get('cam_K')))
            intr = intr.reshape(
                3, 4)[:3, :3] if intr.size == 12 else intr.reshape(3, 3)
            fov_poly = _get_camera_fov_bev_polygon(
                lidar2cam_r, lidar2cam_t, intr, imsize, depth=80.0)
            for i, box in enumerate(boxes):
                in_fov_mask[i] = _box_center_in_cam_fov(
                    box, lidar2cam_r, lidar2cam_t, intr, imsize)

        if fov_poly is not None:
            from matplotlib.patches import Polygon as MplPolygon
            fov_patch = MplPolygon(
                fov_poly, fill=True, alpha=0.15, edgecolor='blue', linewidth=2, label='Camera FOV')
            ax.add_patch(fov_patch)
        else:
            # 无相机标定时，用近似前向 70° 扇形作为参考
            th = np.linspace(-np.pi * 35/180, np.pi * 35/180, 32)
            r = 80
            fov_approx = np.column_stack([r * np.cos(th), r * np.sin(th)])
            from matplotlib.patches import Polygon as MplPolygon
            fov_patch = MplPolygon(fov_approx, fill=True, alpha=0.1,
                                   edgecolor='cyan', linewidth=1.5, label='Approx FOV (70deg)')
            ax.add_patch(fov_patch)

        for i, box in enumerate(boxes):
            corners = self.box_to_corners(box)[:4, :2]
            has_cam = fov_poly is not None and in_fov_mask.size == len(boxes)
            if has_cam:
                color = 'lime' if in_fov_mask[i] else 'red'
            else:
                color = 'gold'  # 无相机标定时无法判断，用金色表示
            poly = Polygon(corners, fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(poly)
            if velocities is not None and i < len(velocities):
                vx, vy = velocities[i][:2]
                ax.arrow(box[0], box[1], vx * 0.5, vy *
                         0.5, head_width=0.3, color=color)

        n_in = int(in_fov_mask.sum()) if (
            in_fov_mask.size == len(boxes)) else -1
        n_out = len(boxes) - n_in if n_in >= 0 else -1
        ax.set_aspect('equal')
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left)')
        ax.legend(loc='upper right')
        fov_str = f'绿=FOV内({n_in}) 红=FOV外({n_out})' if n_in >= 0 else '金=未标定相机'
        plt.title(f'BEV | {fov_str} | {self.v2x_side}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
            print("Saved BEV+FOV:", save_path)
        else:
            plt.show()

    # ===============================
    # 原始 lidar 标注可视化（验证点云是否覆盖所有标注）
    # ===============================

    def visualize_bev_original_labels(self, pts, info, save_path=None):
        """
        加载原始 SPD 数据集 label（label/lidar/{frame_id}.json），在 BEV 上叠加显示。
        与 pkl 中 nuScenes 转换后的 GT 不同，此处为转换前的原始 SPD 标注。
        绿框=框内有点云，红框=框内无点云。
        """
        frame_id = info.get('token', info.get('frame_id', ''))
        if not frame_id:
            print("No frame_id/token in info, cannot load original labels.")
            return

        annos = _load_original_lidar_labels(self.data_root, frame_id)
        if annos is None:
            print(
                f"Original SPD label not found: {self.data_root}/.../label/lidar/{frame_id}.json")
            return

        boxes = _lidar_label_json_to_boxes(annos)
        num_pts = _compute_num_pts_per_box(pts, boxes)
        has_pts = num_pts > 0

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c='gray', alpha=0.5)

        n_has, n_no = int(has_pts.sum()), int((~has_pts).sum())
        for i, box in enumerate(boxes):
            corners = self.box_to_corners(box)[:4, :2]
            color = 'lime' if has_pts[i] else 'red'
            poly = Polygon(corners, fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(poly)
            if num_pts[i] > 0:
                ax.text(box[0], box[1], str(num_pts[i]),
                        fontsize=6, color='darkgreen')

        ax.set_aspect('equal')
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left)')
        plt.title(
            f'原始 SPD label | 绿=有点云({n_has}) 红=无点云({n_no}) | 共{len(boxes)}个框')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
            print("Saved BEV original labels:", save_path)
        else:
            plt.show()

    def visualize_bev_original_vs_gt(self, pts, info, save_path=None):
        """
        对比原始 lidar 标注与 pkl 中的 GT：青=原始标注，红=GT。用于检查转换是否一致。
        """
        frame_id = info.get('token', info.get('frame_id', ''))
        annos = _load_original_lidar_labels(self.data_root, frame_id)
        if annos is None:
            print("Original label not found.")
            return
        boxes_orig = _lidar_label_json_to_boxes(annos)
        boxes_gt = info.get('gt_boxes', np.zeros((0, 7)))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c='gray', alpha=0.5)

        for box in boxes_orig:
            corners = self.box_to_corners(box)[:4, :2]
            poly = Polygon(corners, fill=False,
                           edgecolor='cyan', linewidth=1.5)
            ax.add_patch(poly)
        for box in boxes_gt:
            corners = self.box_to_corners(box)[:4, :2]
            poly = Polygon(corners, fill=False, edgecolor='red',
                           linewidth=1.0, linestyle='--')
            ax.add_patch(poly)

        ax.set_aspect('equal')
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        plt.title(f'青=原始标注({len(boxes_orig)}) 红=GT({len(boxes_gt)})')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
            print("Saved BEV original vs GT:", save_path)
        else:
            plt.show()

    # ===============================
    # 图像 + 点云/框投影（验证是否仅对图像区域标注）
    # ===============================

    def visualize_image_with_projection(self, pts, boxes, info, save_path=None):
        """
        将点云和 3D 框投影到相机图像上。若 GT 框大多在图像内，说明可能是基于图像的标注。
        """
        cams = info.get('cams') or {}
        if not cams and hasattr(self, '_load_cams_for_coop_frame'):
            cam_data = self._load_cams_for_coop_frame(info.get('token', ''))
            if cam_data:
                cams = {'VEHICLE_CAM_FRONT': cam_data}
        if not cams:
            print("No camera info in this sample, skip image projection.")
            return
        cam_name = list(cams.keys())[0]
        cam = cams[cam_name]
        lidar2cam_r = np.array(cam.get('lidar2cam_rotation'))
        lidar2cam_t = np.array(cam.get('lidar2cam_translation'))
        intr = np.array(cam.get('cam_intrinsic')).reshape(3, 3)[:3, :3]
        dp = cam.get('data_path', '')
        img_path = dp if cam.get(
            'data_path_is_absolute') else osp.join(self.data_root, dp)
        if not osp.isfile(img_path):
            print("Image not found:", img_path)
            return
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load image:", img_path)
            return
        h, w = img.shape[:2]
        imsize = (w, h)

        # 投影点云（采样以加速）
        step = max(1, len(pts) // 30000)
        pts_sub = pts[::step]
        uv, depth, valid = _points_lidar_to_img(
            pts_sub, lidar2cam_r, lidar2cam_t, intr)
        in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (
            uv[:, 1] >= 0) & (uv[:, 1] < h) & (depth > 0)
        pts_vis = uv[in_img]
        depth_vis = depth[in_img]
        # 深度着色
        depth_norm = (depth_vis - depth_vis.min()) / \
            (depth_vis.max() - depth_vis.min() + 1e-6)
        colors = plt.cm.jet(depth_norm)[:, :3] * 255
        for i in range(len(pts_vis)):
            cv2.circle(img, (int(pts_vis[i, 0]), int(
                pts_vis[i, 1])), 1, colors[i].tolist(), -1)

        # 投影 3D 框
        for box in boxes:
            corners = self.box_to_corners(box)
            uv_c, depth_c, valid_c = _points_lidar_to_img(
                corners, lidar2cam_r, lidar2cam_t, intr)
            if not np.all(valid_c) or np.any(depth_c <= 0):
                continue
            uv_c = uv_c.astype(np.int32)
            in_img_c = (uv_c[:, 0] >= 0) & (uv_c[:, 0] < w) & (
                uv_c[:, 1] >= 0) & (uv_c[:, 1] < h)
            if np.sum(in_img_c) >= 2:
                pts_2d = uv_c
                for k in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                    cv2.line(img, tuple(pts_2d[k[0]]), tuple(
                        pts_2d[k[1]]), (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, img)
            print("Saved image projection:", save_path)
        else:
            cv2.imshow("Image + projection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def analyze_gt_fov(self, max_frames=None):
        """统计 GT 框在相机 FOV 内/外的数量，验证是否仅对图像区域标注。"""
        print("=== GT vs Camera FOV Analysis ===\n")
        total_in, total_out = 0, 0
        max_frames = self.max_frames if max_frames is None else max_frames
        n_frames = len(self.infos) if (max_frames is None or (isinstance(max_frames, str) and max_frames.lower() == "all")) else min(int(max_frames), len(self.infos))
        for idx in range(n_frames):
            info = self.infos[idx]
            boxes = info.get('gt_boxes', [])
            if len(boxes) == 0:
                continue
            cams = info.get('cams') or {}
            if not cams and hasattr(self, '_load_cams_for_coop_frame'):
                cam_data = self._load_cams_for_coop_frame(
                    info.get('token', ''))
                if cam_data:
                    cams = {'cam': cam_data}
            if not cams:
                print(f"Frame {idx}: no camera info, skip.")
                continue
            cam = list(cams.values())[0]
            lidar2cam_r = np.array(cam.get('lidar2cam_rotation', np.eye(3)))
            lidar2cam_t = np.array(
                cam.get('lidar2cam_translation', np.zeros(3)))
            intr = np.array(cam.get('cam_intrinsic')).reshape(3, 3)[:3, :3]
            imsize = (1920, 1080)
            n_in, n_out = 0, 0
            for box in boxes:
                if _box_center_in_cam_fov(box, lidar2cam_r, lidar2cam_t, intr, imsize):
                    n_in += 1
                else:
                    n_out += 1
            total_in += n_in
            total_out += n_out
            if (n_in + n_out) > 0:
                pct = 100 * n_in / (n_in + n_out)
                print(
                    f"Frame {idx}: in_fov={n_in}, out_fov={n_out}, in_fov%={pct:.1f}%")
        print(
            f"\nTotal: in_fov={total_in}, out_fov={total_out}, in_fov%={100*total_in/(total_in+total_out+1e-6):.1f}%")
        if total_out == 0 and total_in > 0:
            print("=> GT 全部在图像 FOV 内，支持「仅对图像区域标注」的假设。")
        elif total_out > 0:
            print("=> GT 存在 FOV 外目标，点云标注可能为全范围。")

    # ===============================
    # 3D 可视化
    # ===============================

    def visualize_3d(self, pts, boxes, save_path="frame3d.png"):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.2)

        for box in boxes:
            corners = self.box_to_corners(box)

            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]

            for l in lines:
                ax.plot(
                    [corners[l[0], 0], corners[l[1], 0]],
                    [corners[l[0], 1], corners[l[1], 1]],
                    [corners[l[0], 2], corners[l[1], 2]]
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.savefig(save_path)
        plt.close()
        print("Saved 3D image:", save_path)

    # ===============================
    # 速度连续性
    # ===============================

    def check_velocity_continuity(self):

        print("=== Velocity Continuity ===")
        last = None

        for idx, info in enumerate(self.infos):

            vel = info.get('gt_velocity', None)
            if vel is None:
                continue

            if last is not None and len(vel) == len(last):
                diff = np.linalg.norm(vel - last, axis=1).mean()
                print(f'Frame {idx}: mean vel diff = {diff:.3f}')

            last = vel

    # ===============================
    # yaw 连续性
    # ===============================

    def check_velocity_continuity(self):

        print("=== Velocity Continuity ===")
        last = None

        for idx, info in enumerate(self.infos):

            vel = info.get('gt_velocity', None)
            if vel is None:
                continue

            if last is not None and len(vel) == len(last):
                diff = np.linalg.norm(vel - last, axis=1).mean()
                print(f'Frame {idx}: mean vel diff = {diff:.3f}')

            last = vel

    # ===============================
    # GT vs 预测 IoU
    # ===============================

    def check_yaw_continuity(self):

        print("=== Yaw Continuity ===")
        last = None

        for idx, info in enumerate(self.infos):

            boxes = info.get('gt_boxes', None)
            if boxes is None:
                continue

            yaw = limit_period(boxes[:, 6])

            if last is not None and len(yaw) == len(last):
                diff = np.abs(limit_period(yaw - last)).mean()
                print(f'Frame {idx}: mean yaw diff = {diff:.3f}')

            last = yaw

    # ===============================
    # Coop 对齐检查
    # ===============================

    def check_coop_alignment(self, veh_boxes, inf_boxes):

        print("=== Coop Alignment ===")
        for v in veh_boxes:
            for i in inf_boxes:
                dist = np.linalg.norm(v[:3] - i[:3])
                if dist < 1.0:
                    print("Aligned object distance:", dist)

    # ===============================
    # 视频导出
    # ===============================

    def export_video(self, save_path='bev_video.mp4', max_frames=None, fps=10, workers=None,
                     fast=True, downsample=5, draw_text=False, use_processes=True):
        """
        导出 BEV 视频。fast=True 使用 OpenCV 渲染。use_processes=True 用多进程(突破GIL)，否则多线程。
        """
        if max_frames is None:
            max_frames = self.max_frames

        frame_size = (800, 800)
        n_workers = workers if workers is not None else min(8, (os.cpu_count() or 4))

        if max_frames is None or (isinstance(max_frames, str) and max_frames.lower() == "all"):
            frames_range = list(range(len(self.infos)))
        else:
            frames_range = list(range(min(int(max_frames), len(self.infos))))

        if fast:
            task_args = [
                (idx, self.infos[idx], self.data_root, frame_size, downsample, draw_text)
                for idx in frames_range
            ]
        else:
            task_args = [
                (idx, self.infos[idx], self.data_root, frame_size)
                for idx in frames_range
            ]

        results = {}
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with ExecutorClass(max_workers=n_workers) as executor:
            worker_fn = _render_single_bev_frame_cv if fast else _render_single_bev_frame
            futures = {executor.submit(worker_fn, args): args[0]
                       for args in task_args}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, img = future.result()
                    if img is not None:
                        results[idx] = img
                except Exception as e:
                    print(f"Frame {idx} render failed: {e}")
                done += 1
                if done % 10 == 0:
                    print(f"Rendered {done}/{len(frames_range)} frames")

        # 按顺序写入视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
        for idx in frames_range:
            if idx in results:
                video.write(results[idx])
        video.release()
        print(f"Saved BEV video: {save_path} ({len(results)} frames)")

    # ===============================
    # 主运行
    # ===============================

    def run(self,
            frame_idx=0,
            show_bev=True,
            show_3d=False,
            show_bev_fov=False,
            show_image_proj=False,
            show_spd_original_labels=False,
            analyze_fov=False,
            check_continuity=False,
            export_video=False,
            save_dir=None,
            max_frames=None,
            workers=None,
            export_fast=True,
            export_downsample=5,
            export_draw_text=False,
            export_use_processes=True):

        max_frames = max_frames if max_frames is not None else self.max_frames

        if analyze_fov:
            self.analyze_gt_fov(max_frames=max_frames)
            return

        if export_video:
            self.export_video(save_path='bev_video.mp4',
                              max_frames=max_frames,
                              workers=workers,
                              fast=export_fast,
                              downsample=export_downsample,
                              draw_text=export_draw_text,
                              use_processes=export_use_processes)
            return

        info = self.infos[frame_idx]
        lidar_path = info['lidar_path'].replace('.bin', '.pcd')
        pts = self.load_lidar(lidar_path)

        boxes = info.get('gt_boxes', [])
        velocities = info.get('gt_velocity', None)

        print("Frame:", frame_idx)
        print("Num GT:", len(boxes))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            pf = osp.join(save_dir, f"frame_{frame_idx:04d}")
        else:
            pf = None

        if show_bev:
            self.visualize_bev(pts, boxes, velocities)

        if show_bev_fov:
            self.visualize_bev_with_fov(
                pts, boxes, info, velocities,
                save_path=f"{pf}_bev_fov.png" if pf else None
            )

        if show_image_proj:
            self.visualize_image_with_projection(
                pts, boxes, info,
                save_path=f"{pf}_image_proj.png" if pf else None
            )

        if show_spd_original_labels:
            self.visualize_bev_original_labels(
                pts, info,
                save_path=f"{pf}_bev_original_labels.png" if pf else None
            )

        if show_3d:
            self.visualize_3d(
                pts, boxes, save_path=f"{pf}_3d.png" if pf else "frame3d.png")

        if check_continuity:
            self.check_velocity_continuity()
            self.check_yaw_continuity()


def parse_args():
    parser = argparse.ArgumentParser("SPD Visualizer")

    parser.add_argument('--info_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--pred_path', type=str, default=None)

    parser.add_argument('--v2x_side', type=str, default='vehicle')
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save visualization to directory')

    parser.add_argument('--show_bev', action='store_true')
    parser.add_argument('--show_3d', action='store_true')
    parser.add_argument('--show_bev_fov', action='store_true',
                        help='BEV + camera FOV overlay (green=in FOV, red=out FOV)')
    parser.add_argument('--show_image_proj', action='store_true',
                        help='Project points/boxes onto camera image')
    parser.add_argument('--show_spd_original_labels', '--show_original_labels', action='store_true',
                        dest='show_spd_original_labels',
                        help='BEV with original SPD labels from label/lidar/*.json (green=has pts, red=no pts). pkl contains nuScenes-converted labels; this flag shows raw SPD labels.')
    parser.add_argument('--analyze_fov', action='store_true',
                        help='Count GT in/out camera FOV across frames')
    parser.add_argument('--check_continuity', action='store_true')
    parser.add_argument('--export_video', action='store_true')
    parser.add_argument('--max_frames', '--max_frames_str', type=str, default=None,
                        dest='max_frames',
                        help='Max frames for analyze_fov/export_video: int or "all" for all frames')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of threads for export_video (default: min(8, cpu_count))')
    parser.add_argument('--no_fast_export', action='store_true',
                        help='Use matplotlib for export_video (slower, default uses OpenCV)')
    parser.add_argument('--export_downsample', type=int, default=5,
                        help='Point cloud downsample factor for export_video (default: 5)')
    parser.add_argument('--export_draw_text', action='store_true',
                        help='Draw ID/class text on each box in export_video')
    parser.add_argument('--use_threads', action='store_true',
                        help='Use threads instead of processes for export_video (default: processes)')

    return parser.parse_args()


def _parse_max_frames(s):
    """Parse max_frames from str: None/''/'all' -> None (all frames), '100' -> 100."""
    if s is None or s == "":
        return None
    if isinstance(s, str) and s.lower() == "all":
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":

    args = parse_args()
    max_frames = _parse_max_frames(args.max_frames)

    visualizer = SPDVisualizer(
        info_path=args.info_path,
        data_root=args.data_root,
        pred_path=args.pred_path,
        v2x_side=args.v2x_side,
        max_frames=max_frames
    )

    visualizer.run(
        frame_idx=args.frame_idx,
        show_bev=args.show_bev,
        show_3d=args.show_3d,
        show_bev_fov=args.show_bev_fov,
        show_image_proj=args.show_image_proj,
        show_spd_original_labels=args.show_spd_original_labels,
        analyze_fov=args.analyze_fov,
        check_continuity=args.check_continuity,
        export_video=args.export_video,
        save_dir=args.save_dir,
        max_frames=max_frames,
        workers=args.workers,
            export_fast=not args.no_fast_export,
            export_downsample=args.export_downsample,
            export_draw_text=args.export_draw_text,
            export_use_processes=not args.use_threads
    )
