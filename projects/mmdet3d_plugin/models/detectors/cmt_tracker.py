# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from ..dense_heads.track_head_plugin import Instances, RunTimeTracker
from ..modules import CrossAgentSparseInteraction
import mmcv
import os
import torch.nn.functional as F
import numpy as np
from ..modules import SpatialTemporalReasoner, LatentTransformation
from ..modules import pos2posemb3d
from torchvision.ops import sigmoid_focal_loss
from projects.mmdet3d_plugin import SPConvVoxelization


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result


@DETECTORS.register_module()
class CMTCoopTracker(MVXTwoStageDetector):
    """
    CoopTrack
    """

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_backbone=None,
        pts_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        pc_range=None,
        inf_pc_range=None,
        post_center_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=3,
        vehicle_id_list=None,
        runtime_tracker=None,
        gt_iou_threshold=0.0,
        freeze_pts_backbone=False,
        freeze_pts_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
        is_cooperation=False,
        train_track=False,
        read_track_query_file_root=None,
        drop_rate=0,
        save_track_query=False,
        save_track_query_file_root='',
        seq_mode=False,
        batch_size=1,
        spatial_temporal_reason=None,
        motion_prediction_ref_update=True,
        if_update_ego=True,
        train_det=False,
        fp_ratio=0.3,
        random_drop=0.1,
        shuffle=False,
        is_motion=False,
        asso_loss_cfg=None,
        **kwargs
    ):
        super(CMTCoopTracker, self).__init__(
            pts_voxel_layer=None,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            **kwargs)

        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.queue_length = queue_length

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

        self.runtime_tracker = RunTimeTracker(
            **runtime_tracker
        )
        if pts_voxel_layer:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_layer)
        # for test memory
        self.scene_token = None
        self.timestamp = None
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.prev_pos = 0
        self.prev_angle = 0

        self.gt_iou_threshold = gt_iou_threshold
        # self.bev_h removed
        # self.freeze_bev_encoder = freeze_bev_encoder

        # spatial-temporal reasoning
        self.STReasoner = SpatialTemporalReasoner(**spatial_temporal_reason)
        self.hist_len = self.STReasoner.hist_len
        self.fut_len = self.STReasoner.fut_len

        self.motion_prediction_ref_update = motion_prediction_ref_update
        self.if_update_ego = if_update_ego
        self.train_det = train_det
        self.fp_ratio = fp_ratio
        self.random_drop = random_drop
        self.shuffle = shuffle
        self.is_motion = is_motion

        # cross-agent query interaction
        self.is_cooperation = is_cooperation
        self.read_track_query_file_root = read_track_query_file_root

        self.drop_rate = drop_rate

        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root

        self.seq_mode = seq_mode
        self.train_track = train_track
        if self.seq_mode:
            self.batch_size = batch_size
            self.test_flag = False
            # for stream train memory
            self.train_prev_infos = {
                'scene_token': [None] * self.batch_size,
                'prev_timestamp': [None] * self.batch_size,
                'track_instances': [None] * self.batch_size,
                'l2g_r_mat': [None] * self.batch_size,
                'l2g_t': [None] * self.batch_size,
                'prev_pos': [0] * self.batch_size,
                'prev_angle': [0] * self.batch_size,
            }

    def init_weights(self):
        """Initialize model weights."""
        super(CMTCoopTracker, self).init_weights()

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_forecasting_locs=None,
                      gt_forecasting_masks=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      # for coop
                      veh2inf_rt=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        if self.test_flag:  # for interval evaluation
            self.reset_memory()
            self.test_flag = False
        losses = dict()
        if self.seq_mode and self.train_track:
            losses_track = self.forward_track_stream_train(points, gt_bboxes_3d, gt_labels_3d, gt_inds, gt_forecasting_locs, gt_forecasting_masks,
                                                           l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs)
            losses.update(losses_track)
        else:
            pts_feats = self.extract_pts_feat(points)
            outs = self.pts_bbox_head(
                pts_feats, img_feats=None, img_metas=img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses_det = self.pts_bbox_head.loss(*loss_inputs)
            losses.update(losses_det)
        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     # for coop
                     veh2inf_rt=None,
                     **kwargs
                     ):
        """Test function
        """
        # import ipdb;ipdb.set_trace()
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        points = [points] if points is None else points

        points = points[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(
            points, l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs)

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed',
                          'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
        return result

    def reset_memory(self):
        self.train_prev_infos['scene_token'] = [None] * self.batch_size
        self.train_prev_infos['prev_timestamp'] = [None] * self.batch_size
        self.train_prev_infos['track_instances'] = [None] * self.batch_size
        self.train_prev_infos['l2g_r_mat'] = [None] * self.batch_size
        self.train_prev_infos['l2g_t'] = [None] * self.batch_size
        self.train_prev_infos['prev_pos'] = [0] * self.batch_size
        self.train_prev_infos['prev_angle'] = [0] * self.batch_size

    def _ego_motion_compensation(self, ref_pts, l2g_r1, l2g_t1, l2g_r2, l2g_t2):
        device = self.reference_points.weight.device
        if not isinstance(self.pc_range, torch.Tensor):
            pc_range = torch.tensor(self.pc_range, device=device)
        else:
            pc_range = self.pc_range.to(device)

        if ref_pts is None:
            return ref_pts
        from projects.mmdet3d_plugin.models.modules.pf_utils import denormalize, normalize
        ref_pts = denormalize(ref_pts.clone(), pc_range)
        l2g_r1 = l2g_r1.to(device)
        l2g_t1 = l2g_t1.to(device)
        l2g_r2 = l2g_r2.to(device)
        l2g_t2 = l2g_t2.to(device)
        ref_pts = ref_pts @ l2g_r1 + l2g_t1 - l2g_t2

        # Avoid CUBLAS / cusolver errors by transposing
        # l2g_r2 is a rotation matrix, its inverse is its transpose
        g2l_r2 = l2g_r2.transpose(0, 1)

        ref_pts = ref_pts @ g2l_r2
        ref_pts = normalize(ref_pts, pc_range)
        return ref_pts

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device

        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        track_instances.ref_pts = reference_points.clone()
        track_instances.query_embeds = query_embeds.clone()
        track_instances.query_feats = torch.zeros(
            (len(track_instances), self.embed_dims), dtype=torch.float32, device=device)

        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)

        """Current frame information"""
        # classification scores
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # bounding boxes
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)

        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.cache_bboxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_ref_pts = reference_points.clone()
        track_instances.cache_query_embeds = query_embeds.clone()
        track_instances.cache_query_feats = torch.zeros(
            (len(track_instances), self.embed_dims), dtype=torch.float32, device=device)
        track_instances.cache_motion_predictions = torch.zeros_like(
            track_instances.motion_predictions)
        track_instances.cache_motion_feats = torch.zeros_like(query_embeds)

        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)
        # motion features
        track_instances.hist_motion_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)

        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)

        return track_instances

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.reference_points.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(
            tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(
            tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        return track_instances.to(device)

    def load_detection_output_into_cache(self, track_instances: Instances, out):
        """ Load output of the detection head into the track_instances cache (inplace)
        """
        with torch.no_grad():
            track_scores = out['all_cls_scores'][-1,
                                                 :].sigmoid().max(dim=-1).values
        if len(track_instances) != len(track_scores):
            track_instances = track_instances[:len(track_scores)]

        track_instances.cache_scores = track_scores.clone()
        track_instances.cache_logits = out['all_cls_scores'][-1].clone()
        track_instances.cache_query_feats = out['query_feats'][-1].clone()
        track_instances.cache_ref_pts = out['ref_pts'][-1].clone(
        ) if out['ref_pts'].dim() == 3 else out['ref_pts'].clone()
        track_instances.cache_bboxes = out['all_bbox_preds'][-1].clone()
        track_instances.cache_query_embeds = self.query_embedding(
            pos2posemb3d(track_instances.cache_ref_pts))
        return track_instances

    def frame_summarization(self, track_instances, tracking=False):
        """ Load the results after spatial-temporal reasoning into track instances
        """
        # inference mode
        if tracking:
            active_mask = (track_instances.cache_scores >=
                           self.runtime_tracker.record_threshold)
            # print(f"update instance: {track_instances.obj_idxes[active_mask]}")
            # active_mask = (track_instances.cache_scores >= 0.0)
        # training mode
        else:
            track_instances.pred_boxes = track_instances.cache_bboxes.clone()
            track_instances.pred_logits = track_instances.cache_logits.clone()
            track_instances.scores = track_instances.cache_scores.clone()
            track_instances.track_scores = track_instances.cache_scores.clone()
            active_mask = (track_instances.cache_scores >= 0.0)

        track_instances.pred_logits[active_mask] = track_instances.cache_logits[active_mask]
        track_instances.scores[active_mask] = track_instances.cache_scores[active_mask]
        track_instances.track_scores[active_mask] = track_instances.cache_scores[active_mask]
        track_instances.pred_boxes[active_mask] = track_instances.cache_bboxes[active_mask]
        ref_pts = track_instances.ref_pts.clone()
        ref_pts[active_mask] = track_instances.cache_ref_pts[active_mask]
        track_instances.ref_pts = ref_pts
        query_embeds = track_instances.query_embeds.clone()
        query_embeds[active_mask] = track_instances.cache_query_embeds[active_mask]
        track_instances.query_embeds = query_embeds
        query_feats = track_instances.query_feats.clone()
        query_feats[active_mask] = track_instances.cache_query_feats[active_mask]
        track_instances.query_feats = query_feats
        track_instances.motion_predictions[active_mask] = track_instances.cache_motion_predictions[active_mask]

        if self.STReasoner.future_reasoning:
            motion_predictions = track_instances.motion_predictions[active_mask]
            track_instances.fut_xyz[active_mask] = track_instances.ref_pts[active_mask].clone()[
                :, None, :].repeat(1, self.fut_len, 1)
            track_instances.fut_bboxes[active_mask] = track_instances.pred_boxes[active_mask].clone()[
                :, None, :].repeat(1, self.fut_len, 1)
            motion_add = torch.cumsum(
                motion_predictions.clone().detach(), dim=1)
            motion_add_normalized = motion_add.clone()
            motion_add_normalized[...,
                                  0] /= (self.pc_range[3] - self.pc_range[0])
            motion_add_normalized[...,
                                  1] /= (self.pc_range[4] - self.pc_range[1])
            track_instances.fut_xyz[active_mask, :,
                                    0] += motion_add_normalized[..., 0]
            track_instances.fut_xyz[active_mask, :,
                                    1] += motion_add_normalized[..., 1]
            track_instances.fut_bboxes[active_mask, :, 0] += motion_add[..., 0]
            track_instances.fut_bboxes[active_mask, :, 1] += motion_add[..., 1]
        return track_instances

    def _update_track_id_targets(self, track_instances, match_info):
        """Write GT-matched metadata back to track instances for training."""
        if hasattr(track_instances, "matched_gt_idxes"):
            track_instances.matched_gt_idxes.fill_(-1)
        if hasattr(track_instances, "iou"):
            track_instances.iou.zero_()
        if hasattr(track_instances, "track_scores"):
            track_instances.track_scores = track_instances.cache_scores.clone()
        if match_info is None or match_info["query_inds"].numel() == 0:
            return track_instances

        query_inds = match_info["query_inds"].to(
            device=track_instances.cache_scores.device, dtype=torch.long)
        matched_gt_idxes = match_info["matched_gt_idxes"].to(
            device=track_instances.matched_gt_idxes.device, dtype=torch.long)
        obj_idxes = match_info["obj_idxes"].to(
            device=track_instances.obj_idxes.device, dtype=torch.long)
        track_scores = match_info["track_scores"].to(
            device=track_instances.scores.device, dtype=track_instances.scores.dtype)

        track_instances.matched_gt_idxes[query_inds] = matched_gt_idxes
        valid_obj_mask = obj_idxes >= 0
        if valid_obj_mask.any():
            track_instances.obj_idxes[query_inds[valid_obj_mask]
                                      ] = obj_idxes[valid_obj_mask]
        track_instances.scores[query_inds] = track_scores
        if hasattr(track_instances, "track_scores"):
            track_instances.track_scores[query_inds] = match_info["track_scores"].to(
                device=track_instances.track_scores.device,
                dtype=track_instances.track_scores.dtype)
        if hasattr(track_instances, "iou"):
            track_instances.iou[query_inds] = track_instances.iou.new_tensor(
                1.0)
        if hasattr(track_instances, "track_query_mask"):
            track_instances.track_query_mask[query_inds] = True
        return track_instances

    def update_reference_points(self, track_instances, time_delta=None, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction/velocities
        """
        track_instances = self.STReasoner.update_reference_points(
            track_instances, time_delta, use_prediction, tracking)
        return track_instances

    def update_ego(self, track_instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        """
        track_instances = self.STReasoner.update_ego(
            track_instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2)
        return track_instances

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @auto_fp16(apply_to=("points",))
    def _forward_single_frame_train_bs(
        self,
        points,
        img_metas,
        track_instances,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        gt_forecasting_locs=None,
        gt_forecasting_masks=None,
        **kwargs,
    ):
        """
        Perform forward only on one frame. Called in forward_train.
        Only Support BS=1
        """
        bs = len(track_instances)
        for i in range(bs):
            prev_active_track_instances = track_instances[i]

            if prev_active_track_instances is None:
                track_instances[i] = self._generate_empty_tracks()
            else:
                if l2g_r1[i] is not None and l2g_t1[i] is not None:
                    prev_active_track_instances.ref_pts = self._ego_motion_compensation(
                        prev_active_track_instances.ref_pts.clone(), l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i])

                prev_active_track_instances = self.STReasoner.sync_pos_embedding(
                    prev_active_track_instances, self.query_embedding)

                empty_track_instances = self._generate_empty_tracks()
                full_length = len(empty_track_instances)
                active_length = len(prev_active_track_instances)
                if active_length > 0:
                    random_index = torch.randperm(full_length)
                    selected = random_index[:full_length-active_length]
                    empty_track_instances = empty_track_instances[selected]
                out_track_instances = Instances.cat(
                    [empty_track_instances, prev_active_track_instances])
                track_instances[i] = out_track_instances

            if self.shuffle:
                shuffle_index = torch.randperm(len(track_instances[i]))
                track_instances[i] = track_instances[i][shuffle_index]

        for j in range(bs):
            if gt_bboxes_3d is not None and len(gt_bboxes_3d) > j:
                img_metas[j]['gt_bboxes_3d'] = gt_bboxes_3d[j]
            if gt_labels_3d is not None and len(gt_labels_3d) > j:
                img_metas[j]['gt_labels_3d'] = gt_labels_3d[j]
            if gt_inds is not None and len(gt_inds) > j:
                img_metas[j]['gt_inds'] = gt_inds[j]

        pts_feats = self.extract_pts_feat(
            points, img_feats=None, img_metas=img_metas)
        # Call CmtLidarHead natively
        head_outs = self.pts_bbox_head(
            pts_feats, img_metas=img_metas, track_instances=track_instances)

        last_outs = head_outs[0][0]
        output_classes = last_outs['cls_logits']
        output_coords = torch.cat([last_outs['center'], last_outs['height'],
                                  last_outs['dim'], last_outs['rot'], last_outs['vel']], dim=-1)
        last_ref_pts = torch.cat(
            [last_outs['norm_center'][-1], last_outs['norm_height'][-1]], dim=-1)
        query_feats = last_outs['query_feats']
        track_match_bboxes = []
        track_match_logits = []
        for task_out in head_outs:
            task_last = task_out[0]
            track_match_bboxes.append(torch.cat(
                [task_last['center'][-1], task_last['height'][-1],
                 task_last['dim'][-1], task_last['rot'][-1], task_last['vel'][-1]], dim=-1))
            track_match_logits.append(task_last['cls_logits'][-1])

        losses = {}
        out = {'track_instances': []}
        avg_factors = {}

        # Compute base detection losses for the whole batch natively
        det_losses = self.pts_bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, head_outs)
        det_losses = {f"det_{k}": v for k, v in det_losses.items()}

        for j in range(bs):
            cur_loss = dict()
            cur_track_instances = track_instances[j]
            cur_out = {
                'all_cls_scores': output_classes[:, j, :, :].detach(),
                'all_bbox_preds': output_coords[:, j, :, :].detach(),
                'ref_pts': last_ref_pts[j, :, :].detach(),
                'query_feats': query_feats[:, j, :, :].detach(),
            }

            cur_track_instances = self.load_detection_output_into_cache(
                cur_track_instances, cur_out)
            match_info = self.pts_bbox_head.assign_track_targets(
                gt_bboxes_3d[j],
                gt_labels_3d[j],
                [task_bbox[j].detach() for task_bbox in track_match_bboxes],
                [task_logits[j].detach() for task_logits in track_match_logits],
                gt_inds=gt_inds[j] if gt_inds is not None and len(
                    gt_inds) > j else None,
            )
            cur_track_instances = self._update_track_id_targets(
                cur_track_instances, match_info)
            cur_out['track_instances'] = cur_track_instances

            # Base Detection Loss (from pts_bbox_head directly)
            num_tracks = len(
                prev_active_track_instances) if prev_active_track_instances is not None else 0
            new_boxes = cur_track_instances.cache_bboxes[num_tracks:]
            new_logits = cur_track_instances.cache_logits[num_tracks:]

            # The base detection losses (including intermediate layers, dn_loss, and the final layer
            # for both track and new queries) are now computed outside this loop in `det_losses`.

            if not self.is_cooperation:
                # Update tracks
                pass

            inf_instances = None
            if self.is_cooperation:
                inf_dcit = {
                    'query_feats': kwargs['query_feats'][j][0],
                    'query_embeds': kwargs['query_embeds'][j][0],
                    'cache_motion_feats': kwargs['cache_motion_feats'][j][0] if 'cache_motion_feats' in kwargs else None,
                    'ref_pts': kwargs['ref_pts'][j][0],
                    'pred_boxes': kwargs['pred_boxes'][j][0],
                }
                if inf_dcit['query_feats'].shape[0] > 0:
                    inf_dcit = self.crossview_alignment(
                        inf_dcit, veh2inf_rt[j])
                    inf_instances = self._init_inf_tracks(inf_dcit)
                if self.STReasoner.learn_match:
                    mask = cur_track_instances.cache_scores > self.STReasoner.veh_thre
                    veh_boxes = cur_track_instances[mask].cache_bboxes.clone()
                    inf_boxes = inf_instances.cache_bboxes.clone()
                    asso_label = self.STReasoner._gen_asso_label(
                        gt_bboxes_3d[j], inf_boxes, veh_boxes, img_metas[j]['sample_idx'])

            # 3. Spatial-temporal reasoning
            cur_track_instances, affinity = self.STReasoner(
                cur_track_instances, inf_instances)

            if self.is_cooperation:
                pass  # Cooperative loss logic to be added if needed

            if self.STReasoner.history_reasoning:
                pass  # Mem bank loss to be added if needed

            if self.STReasoner.future_reasoning:
                pass  # Forecasting loss to be added if needed

            cur_track_instances = self.frame_summarization(
                cur_track_instances, tracking=False)
            active_mask = self.runtime_tracker.get_active_mask(
                cur_track_instances, training=True)
            cur_track_instances.track_query_mask[active_mask] = True
            active_track_instances = cur_track_instances[active_mask]

            if self.random_drop > 0.0:
                active_track_instances = self._random_drop_tracks(
                    active_track_instances)
            if self.fp_ratio > 0.0:
                active_track_instances = self._add_fp_tracks(
                    cur_track_instances, active_track_instances)
            out['track_instances'].append(active_track_instances)

            for key, value in cur_loss.items():
                if 'loss' not in key:
                    continue
                af_key = key.replace('loss', 'avg_factor')
                avg_factor = cur_loss[af_key]
                if key not in losses:
                    losses[key] = value * avg_factor
                    avg_factors[af_key] = avg_factor
                else:
                    losses[key] = losses[key] + value * avg_factor
                    avg_factors[af_key] = avg_factors[af_key] + avg_factor

        for key, value in losses.items():
            af_key = key.replace('loss', 'avg_factor')
            avg_factor = avg_factors[af_key]
            losses[key] = value / avg_factor

        losses.update(det_losses)
        return out, losses

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(
                track_instances.scores) > drop_probability
            track_instances = track_instances[keep_idxes]
        return track_instances

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        """
        self.fp_ratio is used to control num(add_fp) / num(active)
        """
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(
            active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # randomly select num_fp from inactive_instances
                # fp_indexes = np.random.permutation(len(inactive_instances))
                # fp_indexes = fp_indexes[:num_fp]
                # fp_track_instances = inactive_instances[fp_indexes]

                # v2: select the fps with top scores rather than random selection
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat(
                [active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(
            track_instances[active_index], img_metas, with_mask=with_mask)
        # result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[
            active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict

    def _forward_single_frame_inference(
        self,
        points,
        img_metas,
        track_instances,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        sample_idx=None,
        **kwargs
    ):
        prev_active_track_instances = track_instances
        if prev_active_track_instances is None:
            track_instances = self._generate_empty_tracks()
        else:
            if l2g_r1[0] is not None and l2g_t1[0] is not None:
                prev_active_track_instances.ref_pts = self._ego_motion_compensation(
                    prev_active_track_instances.ref_pts.clone(), l2g_r1[0], l2g_t1[0], l2g_r2[0], l2g_t2[0])

            prev_active_track_instances = self.STReasoner.sync_pos_embedding(
                prev_active_track_instances, self.query_embedding)

            empty_track_instances = self._generate_empty_tracks()
            full_length = len(empty_track_instances)
            active_length = len(prev_active_track_instances)
            if active_length > 0:
                random_index = torch.randperm(full_length)
                selected = random_index[:full_length-active_length]
                empty_track_instances = empty_track_instances[selected]
            out_track_instances = Instances.cat(
                [empty_track_instances, prev_active_track_instances])
            track_instances = out_track_instances

        pts_feats = self.extract_pts_feat(points)
        head_outs = self.pts_bbox_head(
            pts_feats, img_metas=[img_metas[0]], track_instances=track_instances)

        last_outs = head_outs[0][0]
        output_classes = last_outs['cls_logits']
        output_coords = torch.cat([last_outs['center'], last_outs['height'],
                                  last_outs['dim'], last_outs['rot'], last_outs['vel']], dim=-1)
        last_ref_pts = torch.cat(
            [last_outs['norm_center'][-1], last_outs['norm_height'][-1]], dim=-1)
        query_feats = last_outs['query_feats']

        out = {
            "all_cls_scores": output_classes[:, 0, :, :],
            "all_bbox_preds": output_coords[:, 0, :, :],
            "ref_pts": last_ref_pts[0, :, :],
            "query_feats": query_feats[:, 0, :, :],
        }
        track_instances = self.load_detection_output_into_cache(
            track_instances, out)
        out['track_instances'] = track_instances

        # if self.is_motion:
        #     track_instances = self.MotionExtractor(track_instances, img_metas)

        inf_instances = None
        if self.is_cooperation:
            inf_dcit = {
                'query_feats': kwargs['query_feats'][0][0],
                'query_embeds': kwargs['query_embeds'][0][0],
                'cache_motion_feats': kwargs['cache_motion_feats'][0][0] if 'cache_motion_feats' in kwargs else None,
                'ref_pts': kwargs['ref_pts'][0][0],
                'pred_boxes': kwargs['pred_boxes'][0][0],
            }
            if inf_dcit['query_feats'].shape[0] > 0:
                inf_dcit = self.crossview_alignment(inf_dcit, veh2inf_rt[0])
                inf_instances = self._init_inf_tracks(inf_dcit)

        # Spatial-temporal Reasoning
        track_instances, _ = self.STReasoner(
            track_instances, inf_instances, sample_idx)
        track_instances = self.frame_summarization(
            track_instances, tracking=True)

        out['all_cls_scores'][-1] = track_instances.pred_logits
        out['all_bbox_preds'][-1] = track_instances.pred_boxes

        if self.STReasoner.future_reasoning:
            out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
        else:
            out['all_motion_forecasting'] = None

        active_mask = (track_instances.scores > self.runtime_tracker.threshold)
        track_instances = self.runtime_tracker.get_assign_ids(
            track_instances, active_mask)

        out['track_instances'] = track_instances
        active_index = (track_instances.scores >=
                        self.runtime_tracker.output_threshold)
        out.update(self.select_active_track_query(
            track_instances, active_index, img_metas))

        next_instances = self.runtime_tracker.update_active_tracks(
            track_instances, active_mask)
        out["track_instances"] = next_instances
        out.update(self._det_instances2results(out, img_metas))
        out["track_obj_idxes"] = track_instances.obj_idxes

        return out

    def forward_track_stream_train(self,
                                   points=None,
                                   gt_bboxes_3d=None,
                                   gt_labels_3d=None,
                                   gt_inds=None,
                                   gt_forecasting_locs=None,
                                   gt_forecasting_masks=None,
                                   l2g_t=None,
                                   l2g_r_mat=None,
                                   img_metas=None,
                                   timestamp=None,
                                   veh2inf_rt=None,
                                   **kwargs):
        bs = len(points)
        time_delta = [None] * bs
        l2g_r1 = [None] * bs
        l2g_t1 = [None] * bs
        l2g_r2 = [None] * bs
        l2g_t2 = [None] * bs
        for i in range(bs):
            tmp_pos = copy.deepcopy(img_metas[i][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[i][0]['can_bus'][-1])
            if img_metas[i][0]['scene_token'] != self.train_prev_infos['scene_token'][i] or \
                    timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i] > 0.5 or \
                    timestamp[i][0] < self.train_prev_infos['prev_timestamp'][i]:
                self.train_prev_infos['track_instances'][i] = None
                time_delta[i], l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i] = None, None, None, None, None
                img_metas[i][0]['can_bus'][:3] = 0
                img_metas[i][0]['can_bus'][-1] = 0
            else:
                time_delta[i] = timestamp[i][0] - \
                    self.train_prev_infos['prev_timestamp'][i]
                assert time_delta[i] > 0
                l2g_r1[i] = self.train_prev_infos['l2g_r_mat'][i]
                l2g_t1[i] = self.train_prev_infos['l2g_t'][i]
                l2g_r2[i] = l2g_r_mat[i][0]
                l2g_t2[i] = l2g_t[i][0]
                img_metas[i][0]['can_bus'][:3] -= self.train_prev_infos['prev_pos'][i]
                img_metas[i][0]['can_bus'][-1] -= self.train_prev_infos['prev_angle'][i]

            self.train_prev_infos['scene_token'][i] = img_metas[i][0]['scene_token']
            self.train_prev_infos['prev_timestamp'][i] = timestamp[i][0]
            self.train_prev_infos['l2g_r_mat'][i] = l2g_r_mat[i][0]
            self.train_prev_infos['l2g_t'][i] = l2g_t[i][0]
            self.train_prev_infos['prev_pos'][i] = tmp_pos
            self.train_prev_infos['prev_angle'][i] = tmp_angle

        if not self.train_track:
            track_instances = [None for i in range(bs)]
        else:
            track_instances = self.train_prev_infos['track_instances'][:bs]

        points_single = [pts[0] for pts in points]
        img_metas_single = [copy.deepcopy(img_metas[i][0])
                            for i in range(bs)]

        frame_res, losses = self._forward_single_frame_train_bs(
            points_single,
            img_metas_single,
            track_instances,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,

            **kwargs,
        )
        track_instances = frame_res["track_instances"]

        for i in range(bs):
            self.train_prev_infos['track_instances'][i] = track_instances[i].detach_and_clone(
            )
        return losses

    def simple_test_track(
        self,
        points=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        veh2inf_rt=None,
        **kwargs,
    ):
        bs = points.size(0)
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
            or timestamp - self.timestamp > 0.5
            or timestamp < self.timestamp
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            track_instances = None
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            img_metas[0]['can_bus'][:3] = 0
            img_metas[0]['can_bus'][-1] = 0
            self.runtime_tracker.empty()
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
            img_metas[0]['can_bus'][:3] -= self.prev_pos
            img_metas[0]['can_bus'][-1] -= self.prev_angle

        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat
        self.prev_pos = tmp_pos
        self.prev_angle = tmp_angle

        if not self.train_track:
            track_instances = None

        frame_res = self._forward_single_frame_inference(
            points,
            img_metas,
            track_instances,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt,
            img_metas[0]['sample_idx'],
            **kwargs
        )

        track_instances = frame_res["track_instances"]
        self.test_track_instances = track_instances

        results = [dict()]
        get_keys = ["track_bbox_results", "boxes_3d_det", "scores_3d_det", "labels_3d_det",
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        results[0].update({k: frame_res[k]
                          for k in get_keys if k in frame_res})

        if self.save_track_query:
            tensor_to_cpu = torch.zeros(1)
            save_path = os.path.join(
                self.save_track_query_file_root, img_metas[0]['sample_idx'] + '.pkl')
            track_instances = track_instances.to(tensor_to_cpu)
            mmcv.dump(track_instances, save_path)

        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        bboxes_dict = self.pts_bbox_head.bbox_coder.decode(
            bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(
            ), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, pred_dict, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # import pdb;pdb.set_trace()
        cls_score = pred_dict['all_cls_scores'].clone()
        bbox_preds = pred_dict['all_bbox_preds'].clone()
        scores = cls_score[-1].sigmoid().max(dim=-1).values
        obj_idxes = torch.ones_like(scores)
        bbox_dict = dict(
            cls_scores=cls_score[-1],
            bbox_preds=bbox_preds[-1],
            track_scores=scores,
            obj_idxes=obj_idxes,
        )
        bboxes_dict = self.pts_bbox_head.bbox_coder.decode(
            bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )

        return result_dict_det
