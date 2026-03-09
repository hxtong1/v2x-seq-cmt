# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import os

import numpy as np
from os import path as osp

import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.loaders import (
    load_prediction,
    load_gt,
    add_center_dist,
    filter_eval_boxes,
)
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.evaluate import DetectionEval


class _DetectionEvalFromBoxes(DetectionEval):
    """DetectionEval that uses pre-loaded pred_boxes and gt_boxes (same sample_tokens)."""

    def __init__(self, nusc, config, pred_boxes, gt_boxes, meta, output_dir, verbose=False):
        self.nusc = nusc
        self.result_path = ''
        self.eval_set = ''
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.plot_dir = osp.join(self.output_dir, 'plots')
        if not osp.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not osp.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.pred_boxes = pred_boxes
        self.gt_boxes = gt_boxes
        self.sample_tokens = self.gt_boxes.sample_tokens
        self.meta = meta


def _load_gt_from_data_infos(dataset, sample_tokens, box_cls=DetectionBox, verbose=False):
    """Load GT from dataset.data_infos (pkl) with correct gt_velocity, bypassing nusc.box_velocity.
    nusc.box_velocity 对 prev/next 链错误的 ann 会返回极大值，改用 pkl 的 gt_velocity（lidar 转 global）。
    """
    from pyquaternion import Quaternion
    import numpy as np

    token2info = {info['token']
        : info for info in dataset.data_infos if 'token' in info}
    all_annotations = EvalBoxes()
    for sample_token in sample_tokens:
        info = token2info.get(sample_token)
        if info is None or 'gt_boxes' not in info:
            continue
        gt_boxes = np.array(info['gt_boxes'])
        gt_names = np.array(info['gt_names'])
        gt_velocity = np.array(
            info.get('gt_velocity', np.zeros((len(gt_boxes), 2))))
        if len(gt_velocity) != len(gt_boxes):
            gt_velocity = np.zeros((len(gt_boxes), 2))
        l2e_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_t = np.array(info['lidar2ego_translation'])
        e2g_r = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_t = np.array(info['ego2global_translation'])
        sample_boxes = []
        for i in range(len(gt_boxes)):
            cx, cy, cz = gt_boxes[i, :3]
            w, l, h = gt_boxes[i, 3:6]
            rot = gt_boxes[i, 6]
            vx, vy = float(gt_velocity[i, 0]), float(gt_velocity[i, 1])
            vel_lidar = np.array([vx, vy, 0.0])
            vel_global = (e2g_r @ l2e_r @ vel_lidar)[:2]
            center_ego = l2e_r @ np.array([cx, cy, cz]) + l2e_t
            center_global = (e2g_r @ center_ego + e2g_t).tolist()
            q_lidar = Quaternion(axis=[0, 0, 1], angle=float(rot))
            q_global = Quaternion(e2g_r) * Quaternion(l2e_r) * q_lidar
            size_wlh = [float(w), float(l), float(h)]
            detection_name = str(gt_names[i]) if gt_names[i] in (
                'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                'pedestrian', 'bicycle', 'motorcycle', 'barrier', 'traffic_cone') else None
            if detection_name is None:
                continue
            nl = info.get('num_lidar_pts', np.zeros(len(gt_boxes)))
            nr = info.get('num_radar_pts', np.zeros(len(gt_boxes)))
            num_pts = int(nl[i] if hasattr(nl, '__getitem__') else 0) + \
                int(nr[i] if hasattr(nr, '__getitem__') else 0)
            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=center_global,
                    size=size_wlh,
                    rotation=q_global.elements.tolist(),
                    velocity=vel_global.tolist(),
                    num_pts=num_pts,
                    detection_name=detection_name,
                    detection_score=-1.0,
                    attribute_name='',
                )
            )
        if sample_boxes:
            all_annotations.add_boxes(sample_token, sample_boxes)
    if verbose:
        print('Loaded GT from data_infos for {} samples.'.format(
            len(all_annotations.sample_tokens)))
    return all_annotations


def _load_gt_for_sample_tokens(nusc, sample_tokens, box_cls=DetectionBox, verbose=False):
    """Load GT from nusc for the given sample_tokens (no split filter).

    Use this when the dataset has its own split (e.g. SPD v1.0-trainval) so that
    load_gt(nusc, 'val') returns no samples. Returns EvalBoxes for only the
    tokens that exist in nusc.
    """
    attribute_map = {a['token']: a['name'] for a in nusc.attribute}
    all_annotations = EvalBoxes()
    for sample_token in sample_tokens:
        try:
            sample = nusc.get('sample', sample_token)
        except KeyError:
            continue
        sample_annotation_tokens = sample['anns']
        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = nusc.get(
                'sample_annotation', sample_annotation_token)
            cat_name = sample_annotation['category_name']
            detection_name = category_to_detection_name(cat_name)
            if detection_name is None:
                # SPD and similar use short names (car, pedestrian); accept them.
                if cat_name in ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                                'pedestrian', 'bicycle', 'motorcycle', 'barrier', 'traffic_cone'):
                    detection_name = cat_name
                else:
                    continue
            attr_tokens = sample_annotation['attribute_tokens']
            attr_count = len(attr_tokens)
            if attr_count == 0:
                attribute_name = ''
            elif attr_count == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                attribute_name = attribute_map[attr_tokens[0]]
            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    num_pts=sample_annotation['num_lidar_pts'] +
                    sample_annotation['num_radar_pts'],
                    detection_name=detection_name,
                    detection_score=-1.0,
                    attribute_name=attribute_name,
                )
            )
        all_annotations.add_boxes(sample_token, sample_boxes)
    if verbose:
        print('Loaded GT for {} samples (from {} requested).'.format(
            len(all_annotations.sample_tokens), len(sample_tokens)))
    return all_annotations


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, *args, return_gt_info=False, **kwargs):
        super(CustomNuScenesDataset, self).__init__(*args, **kwargs)
        self.return_gt_info = return_gt_info

    def get_ann_info(self, index):
        """Override to ensure gt_labels_3d are 0..K-1 aligned with self.CLASSES.

        When using 3-class (or any subset) head with tasks like [car], [pedestrian],
        [bicycle], the head matches gt_labels_3d by (task_id, flag): task0 expects
        label 0, task1 expects 1, task2 expects 2. If the parent or pkl provides
        labels from another scheme (e.g. nuScenes 10-class: car=0, pedestrian=6,
        bicycle=7), task1/task2 would get num_total_pos=0 -> bbox loss div by zero
        and grad_norm nan. So we always recompute gt_labels_3d from gt_names using
        self.CLASSES and drop boxes not in self.CLASSES.
        """
        annos = super(CustomNuScenesDataset, self).get_ann_info(index)
        if 'gt_names' not in annos:
            return annos
        gt_names = annos['gt_names']
        # Recompute labels so that label = self.CLASSES.index(name) -> 0..K-1
        keep = []
        new_labels = []
        new_names = []
        for i, name in enumerate(gt_names):
            if name in self.CLASSES:
                keep.append(i)
                new_labels.append(self.CLASSES.index(name))
                new_names.append(name)
        if not keep:
            # No box in CLASSES: return empty but consistent structure
            annos['gt_labels_3d'] = np.array([], dtype=np.int64)
            if 'gt_names' in annos:
                annos['gt_names'] = np.array([], dtype=object)
            if hasattr(annos['gt_bboxes_3d'], 'tensor'):
                from mmdet3d.core.bbox import LiDARInstance3DBoxes
                box_dim = annos['gt_bboxes_3d'].tensor.shape[1]
                annos['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                    np.zeros((0, box_dim), dtype=np.float32), box_dim=box_dim)
            else:
                annos['gt_bboxes_3d'] = np.zeros(
                    (0, annos['gt_bboxes_3d'].shape[1]), dtype=np.float32)
            return annos
        keep = np.asarray(keep)
        annos['gt_labels_3d'] = np.array(new_labels, dtype=np.int64)
        annos['gt_names'] = np.array(new_names, dtype=object)
        annos['gt_bboxes_3d'] = annos['gt_bboxes_3d'][keep]
        for k in list(annos.keys()):
            if k in ('gt_bboxes_3d', 'gt_labels_3d', 'gt_names'):
                continue
            v = annos[k]
            if isinstance(v, np.ndarray) and len(v) == len(gt_names):
                annos[k] = v[keep]
        return annos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        lidar_path = info['lidar_path']
        pts_filename = os.path.splitext(lidar_path)[0] + '.pcd'
        # Resolve sweep data_path: mmdet3d LoadPointsFromMultiSweeps expects loadable path
        sweeps = []
        for sw in info['sweeps']:
            s = dict(sw)
            dp = s.get('data_path', s.get('lidar_path', ''))
            if dp and not osp.isabs(dp):
                s['data_path'] = osp.join(self.data_root, dp)
            elif 'data_path' not in s and 'lidar_path' in s:
                s['data_path'] = osp.join(self.data_root, s['lidar_path'])
            sweeps.append(s)
        if 'token_inf' in info:
            token_inf = info['token_inf']
        else:
            token_inf = -1
        input_dict = dict(
            sample_idx=info['token'],
            sample_idx_inf=token_inf,
            pts_filename=pts_filename,
            sweeps=sweeps,
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            img_sweeps=None if 'img_sweeps' not in info else info['img_sweeps'],
            radar_info=None if 'radars' not in info else info['radars']
        )
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32)))

        if 'VehLidar2InfLidar_rotation' in info:
            veh2inf_r = info['VehLidar2InfLidar_rotation']
        if 'VehLidar2InfLidar_translation' in info:
            veh2inf_t = info['VehLidar2InfLidar_translation']

        veh2inf_rt = np.eye(4)
        if 'VehLidar2InfLidar_rotation' in info and 'VehLidar2InfLidar_translation' in info:
            veh2inf_rt[:3, :3] = veh2inf_r
            veh2inf_rt[:3, 3] = veh2inf_t
            veh2inf_rt = veh2inf_rt.T
        input_dict.update(dict(veh2inf_rt=veh2inf_rt.astype(np.float32)))

        if self.return_gt_info:
            input_dict['info'] = info

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def _evaluate_single_fallback(self, result_name):
        """Return zero metrics when official nuScenes eval is not applicable."""
        metric_prefix = f'{result_name}_NuScenes'
        detail = dict()
        for name in self.CLASSES:
            for k in ['0.5', '1.0', '2.0', '4.0']:
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = 0.0
            for err_k in ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']:
                detail['{}/{}_{}'.format(metric_prefix, name, err_k)] = 0.0
        for err_k, label in self.ErrNameMapping.items():
            detail['{}/{}'.format(metric_prefix, label)] = 0.0
        detail['{}/NDS'.format(metric_prefix)] = 0.0
        detail['{}/mAP'.format(metric_prefix)] = 0.0
        return detail

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation with custom split: align pred and gt sample_tokens.

        When using a custom val split (e.g. V2X-Seq-SPD), prediction sample_tokens
        may not match the official nuScenes val split. We filter both to the
        intersection and run evaluation on that subset.
        """
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        try:
            nusc = NuScenes(
                version=self.version, dataroot=self.data_root, verbose=False)
        except Exception:
            return self._evaluate_single_fallback(result_name)

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        eval_set = eval_set_map.get(self.version, 'val')

        try:
            pred_boxes, meta = load_prediction(
                result_path,
                self.eval_detection_configs.max_boxes_per_sample,
                DetectionBox,
                verbose=False)
            gt_boxes_full = load_gt(
                nusc, eval_set, DetectionBox, verbose=False)
        except Exception:
            return self._evaluate_single_fallback(result_name)

        pred_tokens = set(pred_boxes.sample_tokens)
        gt_tokens = set(gt_boxes_full.sample_tokens)
        common = pred_tokens & gt_tokens

        # If no overlap with official val split (e.g. SPD with own scene names),
        # load GT from nusc for pred sample_tokens so evaluation still runs.
        if not common:
            gt_boxes_by_tokens = _load_gt_for_sample_tokens(
                nusc, list(pred_tokens), DetectionBox, verbose=False)
            if gt_boxes_by_tokens.sample_tokens:
                common = set(gt_boxes_by_tokens.sample_tokens)
                gt_boxes_full = gt_boxes_by_tokens
                if logger is not None:
                    logger.info(
                        'Using GT loaded by pred sample_tokens (%d samples) '
                        'instead of official %s.', len(common), eval_set)
            else:
                if logger is not None:
                    logger.warning(
                        'Custom split has no sample_tokens in common with '
                        'nuScenes DB. Skipping NuScenes metrics.')
                return self._evaluate_single_fallback(result_name)

        # 优先用 data_infos (pkl) 的 GT，避免 nusc.box_velocity 对 prev/next 链错误 ann 返回极大 velocity
        if hasattr(self, 'data_infos') and self.data_infos and len(common) > 0:
            gt_from_pkl = _load_gt_from_data_infos(
                self, list(common), DetectionBox, verbose=False)
            if gt_from_pkl.sample_tokens:
                gt_boxes_full = gt_from_pkl

        # Filter to common sample_tokens so pred and gt match
        pred_filtered = EvalBoxes()
        gt_filtered = EvalBoxes()
        for tok in common:
            pred_filtered.add_boxes(tok, pred_boxes[tok])
            gt_filtered.add_boxes(tok, gt_boxes_full[tok])

        # Keep nuScenes eval config aligned with dataset classes from config.
        # This controls both evaluated labels and "Per-class results" printing.
        eval_detection_configs = self.eval_detection_configs
        orig_class_names = eval_detection_configs.class_names
        orig_class_range = eval_detection_configs.class_range
        eval_detection_configs.class_names = list(self.CLASSES)
        eval_detection_configs.class_range = {
            k: v for k, v in orig_class_range.items() if k in self.CLASSES
        }

        try:
            add_center_dist(nusc, pred_filtered)
            add_center_dist(nusc, gt_filtered)
            pred_filtered = filter_eval_boxes(
                nusc, pred_filtered,
                eval_detection_configs.class_range, verbose=False)
            gt_filtered = filter_eval_boxes(
                nusc, gt_filtered,
                eval_detection_configs.class_range, verbose=False)

            # Use a custom evaluator that takes pre-filtered boxes (no load/assert)
            nusc_eval = _DetectionEvalFromBoxes(
                nusc=nusc,
                config=eval_detection_configs,
                pred_boxes=pred_filtered,
                gt_boxes=gt_filtered,
                meta=meta,
                output_dir=output_dir,
                verbose=False)
            nusc_eval.main(render_curves=False)
        finally:
            eval_detection_configs.class_names = orig_class_names
            eval_detection_configs.class_range = orig_class_range

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
