# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from bevformer (https://github.com/fundamentalvision/BEVFormer)        #
# ---------------------------------------------------------------------------------#

import copy
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version, print_log
from einops import rearrange
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from collections import defaultdict
from mmdet.models.utils import build_transformer
from .cmt_head import (SeparateTaskHead, CmtLidarHead, LayerNormFunction, GroupLayerNorm1d)

@HEADS.register_module()
class LIDARBEVFormerTrackHead(CmtLidarHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 in_channels,
                 num_query=900,
                 hidden_dim=128,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=120,
                 bev_w=120,

                 sep_head_init_bias=-0.85,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),
                 positional_encoding=None,
                 **kwargs):

        super(LIDARBEVFormerTrackHead, self).__init__(in_channels, num_query, hidden_dim, with_box_refine, as_two_stage, transformer, bbox_coder, num_cls_fcs, code_weights, bev_h, bev_w, sep_head_init_bias, depth_num, norm_bbox, downsample_scale, scalar, noise_scale, noise_trans, dn_weight, split, train_cfg, test_cfg, common_heads, positional_encoding, **kwargs)
        self.debug_stats_interval = int(kwargs.pop('debug_stats_interval', 0))
        self._debug_forward_step = 0
        # Keep option to disable decoder-side iterative reference update.
        # When disabled, head performs CMT-style regression on fixed reference.
        self.use_decoder_inter_ref = bool(kwargs.pop('use_decoder_inter_ref', False))
        # Use inter-level references only when decoder-side update is enabled.
        self.use_inter_reference = bool(kwargs.pop('use_inter_reference', True))
        # Legacy knobs kept for config compatibility; CMT-style update path below
        # does not apply extra y-axis constraints/clipping.
        self.lock_ref_y_to_init = bool(kwargs.pop('lock_ref_y_to_init', False))
        self.center_y_use_init_reference = bool(
            kwargs.pop('center_y_use_init_reference', False))
        self.center_y_delta_scale = float(
            kwargs.pop('center_y_delta_scale', kwargs.pop('center_y_pre_scale', 1.0)))
        self.reg_pre_sigmoid_clip = kwargs.pop('reg_pre_sigmoid_clip', None)
        self.ref_y_blend = float(kwargs.pop('ref_y_blend', 1.0))
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.det_task_ids = []
        class_offset = 0
        for n_cls in self.num_classes:
            self.det_task_ids.append(list(range(class_offset, class_offset + n_cls)))
            class_offset += n_cls
        # Keep one classification branch over all classes for compatibility
        # with downstream detector/criterion usage.
        self.task_num_classes = [sum(self.num_classes)]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.sep_head_init_bias = sep_head_init_bias
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        self.transformer = build_transformer(transformer)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        # Base embedding dim used by transformer decoder heads.
        self.embed_dims = getattr(self.transformer, 'embed_dims', hidden_dim)

        self.with_box_refine = with_box_refine

        assert as_two_stage is False, 'as_two_stage is not supported yet.'
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.common_heads = common_heads
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        # self.tasks_heads = nn.ModuleList()

        # super(LIDARBEVFormerTrackHead, self).__init__(
        #     *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.ref_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.ref_points.weight.data, 0, 1)

        if positional_encoding is None:
            positional_encoding = dict(
                type='LearnedPositionalEncoding',
                num_feats=hidden_dim // 2,
                row_num_embed=bev_h,
                col_num_embed=bev_w,
            )
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        self._init_layers()

    @staticmethod
    def _format_tensor_stats(name, tensor):
        tensor = tensor.detach()
        if tensor.numel() == 0:
            return f"{name}=empty"
        finite_mask = torch.isfinite(tensor)
        if not finite_mask.any():
            return f"{name}=all_non_finite"
        values = tensor[finite_mask]
        mean_v = float(values.mean())
        std_v = float(values.std(unbiased=False))
        min_v = float(values.min())
        max_v = float(values.max())
        return (
            f"{name}[mean={mean_v:.4f},std={std_v:.4f},"
            f"min={min_v:.4f},max={max_v:.4f}]"
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        heads = copy.deepcopy(self.common_heads)
        heads.update(dict(cls_logits=(self.task_num_classes[0], 2)))
        # CMT-lidar style: one grouped SeparateTaskHead outputs all decoder
        # layers directly from `hs` with shape [num_layers, bs, num_query, C].
        self.sep_head = SeparateTaskHead(
            in_channels=self.embed_dims,
            heads=heads,
            groups=num_pred,
            init_bias=self.sep_head_init_bias,
        )
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        super(LIDARBEVFormerTrackHead, self).init_weights()
        nn.init.uniform(self.ref_points.weight.data, 0, 1)

        self.transformer.init_weights()
        self.sep_head.init_weights()

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        # Compatible with both [num_query, 3] and [bs, num_query, 3].
        if reference_points.dim() == 3:
            base_reference_points = reference_points[0]
        else:
            base_reference_points = reference_points

        has_dn_targets = (
            img_metas is not None
            and len(img_metas) > 0
            and all(('gt_bboxes_3d' in img_meta and 'gt_labels_3d' in img_meta) for img_meta in img_metas)
        )

        if self.training and has_dn_targets:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center,
                                 img_meta['gt_bboxes_3d']._data.tensor[:, 3:]), dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i)
                                  for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(
                groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(
                groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (
                    known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (
                    known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (
                    known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(
                base_reference_points.device)
            padded_reference_points = torch.cat(
                [padding_bbox, base_reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat(
                    [torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    base_reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(
                base_reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad *
                              (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad *
                              (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad *
                              (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad *
                              (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            if reference_points.dim() == 3:
                padded_reference_points = reference_points
            else:
                padded_reference_points = base_reference_points.unsqueeze(
                    0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed, bev_pos

    @auto_fp16(apply_to=('mlvl_feats', 'bev_embed', 'query_feats', 'query_embeds', 'ref_points'))
    def forward(self,
                mlvl_feats=None,
                bev_embed=None,
                query_feats=None,
                query_embeds=None,
                ref_points=None,
                img_metas=None,
                prev_bev=None,
                **kwargs):
        """Unified forward wrapper for compatibility.

        This head is used by cooptrack via `get_bev_features/get_detections`,
        but exposing `forward` makes BaseModule usage and quick smoke tests easier.
        """
        if bev_embed is None:
            assert mlvl_feats is not None, 'Either `bev_embed` or `mlvl_feats` must be provided.'
            bev_embed, _ = self.get_bev_features(
                mlvl_feats=mlvl_feats, img_metas=img_metas, prev_bev=prev_bev)

        # Accept both [bev_hw, bs, c] and [bs, bev_hw, c].
        if bev_embed.dim() == 3 and bev_embed.shape[0] != self.bev_h * self.bev_w \
                and bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        bs = bev_embed.shape[1]
        if ref_points is None:
            ref_points = self.ref_points.weight.unsqueeze(0).repeat(bs, 1, 1)
        if query_feats is None:
            query_feats = bev_embed.new_zeros(
                bs, self.num_query, self.embed_dims)
        if query_embeds is None:
            query_embeds = query_feats

        return self.get_detections(
            bev_embed=bev_embed,
            query_feats=query_feats,
            query_embeds=query_embeds,
            ref_points=ref_points,
            img_metas=img_metas,
        )

    def get_detections(
        self,
        bev_embed,
        query_feats,
        query_embeds,
        ref_points,
        img_metas=None,
    ):
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        reference_points = ref_points
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            bev_embed.shape[1], reference_points, img_metas)
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            query_feats,
            query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points,
            reg_branches=None,
            cls_branches=None,
            img_metas=img_metas,
        )
        # [num_layers, num_query, bs, embed_dims] -> [num_layers, bs, num_query, embed_dims]
        hs = torch.nan_to_num(hs)
        hs = hs.permute(0, 2, 1, 3)
        outs = self.sep_head(hs)
        outputs_class = outs['cls_logits']
        log_this_step = (
            self.debug_stats_interval > 0
            and (self._debug_forward_step % self.debug_stats_interval == 0)
        )
        selected_levels = {0, hs.shape[0] - 1}
        reference = inverse_sigmoid(init_reference.clamp(min=1e-5, max=1 - 1e-5))
        reference = reference.unsqueeze(0).expand(hs.shape[0], -1, -1, -1)
        center_pre_sigmoid = outs['center'] + reference[..., 0:2]
        height_pre_sigmoid = outs['height'] + reference[..., 2:3]
        center = center_pre_sigmoid.sigmoid()
        height = height_pre_sigmoid.sigmoid()
        last_ref_points_all = torch.cat([center, height], dim=-1)
        last_ref_points = last_ref_points_all[-1]

        # [cx, cy, log(w), log(l), cz, log(h), sin, cos, vx, vy]
        outputs_coord = torch.cat(
            [center, outs['dim'][..., 0:2], height, outs['dim'][..., 2:3], outs['rot'], outs['vel']],
            dim=-1,
        )
        # Denormalize center to metric space.
        outputs_coord[..., 0:1] = outputs_coord[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        outputs_coord[..., 1:2] = outputs_coord[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        outputs_coord[..., 4:5] = outputs_coord[..., 4:5] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        if log_this_step:
            for lvl in sorted(selected_levels):
                debug_msg = " ".join([
                    f"[det_debug] step={self._debug_forward_step}",
                    f"lvl={lvl}",
                    self._format_tensor_stats("ref_norm", reference[lvl].sigmoid()),
                    self._format_tensor_stats("ref_y_norm", reference[lvl].sigmoid()[..., 1:2]),
                    self._format_tensor_stats("delta_y", center[lvl][..., 1:2]),
                    self._format_tensor_stats("center_pre", center_pre_sigmoid[lvl]),
                    self._format_tensor_stats("center_post", center[lvl]),
                    self._format_tensor_stats("height_pre", height_pre_sigmoid[lvl]),
                    self._format_tensor_stats("height_post", height[lvl]),
                    self._format_tensor_stats("bbox_x", outputs_coord[lvl][..., 0:1]),
                    self._format_tensor_stats("bbox_y", outputs_coord[lvl][..., 1:2]),
                    self._format_tensor_stats("bbox_z", outputs_coord[lvl][..., 4:5]),
                ])
                print_log(debug_msg, logger="mmdet")
        if self.debug_stats_interval > 0:
            self._debug_forward_step += 1
        outputs_classes = outputs_class
        outputs_coords = outputs_coord
        # outputs_trajs = torch.stack(outputs_trajs)
        # last_ref_points = inverse_sigmoid(last_ref_points)

        dn_outputs_classes = None
        dn_outputs_coords = None
        dn_last_ref_points = None
        dn_query_feats = None
        if mask_dict and mask_dict.get('pad_size', 0) > 0:
            pad_size = mask_dict['pad_size']
            # Split denoising queries (prefix) from matching queries.
            dn_outputs_classes = outputs_classes[:, :, :pad_size, :]
            dn_outputs_coords = outputs_coords[:, :, :pad_size, :]
            dn_last_ref_points = last_ref_points[:, :pad_size, :]
            dn_query_feats = hs[:, :, :pad_size, :]

            outputs_classes = outputs_classes[:, :, pad_size:, :]
            outputs_coords = outputs_coords[:, :, pad_size:, :]
            last_ref_points = last_ref_points[:, pad_size:, :]
            hs = hs[:, :, pad_size:, :]

        outs = {
            # [num_layers, bs, num_query, num_cls]
            'all_cls_scores': outputs_classes,
            # [num_layers, bs, num_query, num_dim]
            'all_bbox_preds': outputs_coords,
            # Task-view outputs over global shared queries.
            'all_task_cls_scores': [outputs_classes[..., task_ids] for task_ids in self.det_task_ids],
            'all_task_bbox_preds': [outputs_coords.clone() for _ in self.det_task_ids],
            'task_ids': self.det_task_ids,
            # 'all_past_traj_preds': outputs_trajs,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'last_ref_points': last_ref_points,  # [bs, num_query, 3]
            'query_feats': hs,  # [num_layers, bs, num_query, embed_dims]
        }
        if dn_outputs_classes is not None:
            # Build task-wise dn_mask_dict to align global labels with each task's
            # local class space, same idea as CmtLidarHead.
            task_dn_mask_dicts = []
            if len(self.class_names) > 1:
                flag = 0
                for class_name in self.class_names:
                    task_mask_dict = copy.deepcopy(mask_dict)
                    known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                    known_labels_raw = task_mask_dict['known_labels_raw']

                    new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(
                        known_lbs_bboxes_label.shape)
                    new_lbs_bboxes_label[:] = len(class_name)
                    new_labels_raw = known_labels_raw.new_zeros(
                        known_labels_raw.shape)
                    new_labels_raw[:] = len(class_name)

                    task_masks = [
                        torch.where(known_lbs_bboxes_label ==
                                    class_name.index(i) + flag)
                        for i in class_name
                    ]
                    task_masks_raw = [
                        torch.where(known_labels_raw ==
                                    class_name.index(i) + flag)
                        for i in class_name
                    ]
                    for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                        new_lbs_bboxes_label[task_mask] = class_name.index(
                            cname)
                        new_labels_raw[task_mask_raw] = class_name.index(cname)

                    task_mask_dict['known_lbs_bboxes'] = (
                        new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                    task_mask_dict['known_labels_raw'] = new_labels_raw
                    task_dn_mask_dicts.append(task_mask_dict)
                    flag += len(class_name)
            else:
                task_dn_mask_dicts = [mask_dict]

            outs.update({
                'dn_all_cls_scores': dn_outputs_classes,
                'dn_all_bbox_preds': dn_outputs_coords,
                'dn_all_task_cls_scores': [dn_outputs_classes[..., task_ids] for task_ids in self.det_task_ids],
                'dn_all_task_bbox_preds': [dn_outputs_coords.clone() for _ in self.det_task_ids],
                'dn_last_ref_points': dn_last_ref_points,
                'dn_query_feats': dn_query_feats,
                'dn_mask_dict': task_dn_mask_dicts,
            })
        return outs

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:

            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(
                bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(
                assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
        # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                        num_classes,
                                        dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0

            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks\
            = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)

        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum(
                (inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum(
                (inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append(
                [labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])

        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)

    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox 
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * \
            bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx]
                                    for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx]
                                    for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task,
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']

        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()

        if task_mask_sum > 0:
            # pred_logits = pred_logits[task_mask]
            # known_labels = known_labels[task_mask]
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split

        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * \
            bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        loss_bbox = self.loss_bbox(
            pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_tgt)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            # loss_cls = loss_cls * 0.0
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_decoder = preds_dicts[0][0]['center'].shape[0]
        all_pred_bboxes, all_pred_logits = collections.defaultdict(
            list), collections.defaultdict(list)

        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['center'][dec_id], preds_dict[0]['height'][dec_id],
                     preds_dict[0]['dim'][dec_id], preds_dict[0]['rot'][dec_id],
                     preds_dict[0]['vel'][dec_id]),
                    dim=-1
                )
                all_pred_bboxes[dec_id].append(pred_bbox)
                all_pred_logits[dec_id].append(
                    preds_dict[0]['cls_logits'][dec_id])
        all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
        all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

        loss_cls, loss_bbox = multi_apply(
            self.loss_single, all_pred_bboxes, all_pred_logits,
            [gt_bboxes_3d for _ in range(num_decoder)],
            [gt_labels_3d for _ in range(num_decoder)],
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_bbox'] = loss_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                           loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        dn_pred_bboxes, dn_pred_logits = collections.defaultdict(
            list), collections.defaultdict(list)
        dn_mask_dicts = collections.defaultdict(list)
        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['dn_center'][dec_id], preds_dict[0]['dn_height'][dec_id],
                     preds_dict[0]['dn_dim'][dec_id], preds_dict[0]['dn_rot'][dec_id],
                     preds_dict[0]['dn_vel'][dec_id]),
                    dim=-1
                )
                dn_pred_bboxes[dec_id].append(pred_bbox)
                dn_pred_logits[dec_id].append(
                    preds_dict[0]['dn_cls_logits'][dec_id])
                dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
        dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
        dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
        dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
        dn_loss_cls, dn_loss_bbox = multi_apply(
            self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
        )

        loss_dict['dn_loss_cls'] = dn_loss_cls[-1]
        loss_dict['dn_loss_bbox'] = dn_loss_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                           dn_loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
