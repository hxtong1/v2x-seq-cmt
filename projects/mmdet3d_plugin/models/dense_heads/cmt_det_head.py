# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from distutils.command.build import build
import enum
from turtle import down
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import NormedLinear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from einops import rearrange
import collections
from .cmt_head import CmtHead, CmtLidarHead
from functools import reduce
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C//groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)


@HEADS.register_module()
class SeparateTaskHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 groups=1,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateTaskHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.groups = groups
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv1d(
                        c_in * groups,
                        head_conv * groups,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        groups=groups,
                        bias=False),
                    GroupLayerNorm1d(head_conv * groups, groups=groups),
                    nn.ReLU(inplace=True)
                ])
                c_in = head_conv

            conv_layers.append(
                nn.Conv1d(
                    head_conv * groups,
                    classes * groups,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    groups=groups,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv1d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'cls_logits':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [N, B, query, C].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [N, B, query, 2].
                -height (torch.Tensor): Height value with the \
                    shape of [N, B, query, 1].
                -dim (torch.Tensor): Size value with the shape \
                    of [N, B, query, 3].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [N, B, query, 2].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [N, B, query, 2].
        """
        N, B, query_num, c1 = x.shape
        x = rearrange(x, "n b q c -> b (n c) q")
        ret_dict = dict()
        
        for head in self.heads:
             head_output = self.__getattr__(head)(x)
             ret_dict[head] = rearrange(head_output, "b (n c) q -> n b q c", n=N)

        return ret_dict


@HEADS.register_module()
class CmtDetHead(CmtLidarHead):

    def __init__(self,
                 *args,
                 transformer=None,
                 bev_h=120,
                 bev_w=120,
                 **kwargs):
        super(CmtDetHead, self).__init__(
            
            transformer=transformer, 
            **kwargs)
        
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        # transformer
        self.transformer = build_transformer(transformer)

    def init_weights(self):
        super(CmtDetHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base

    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, _, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer(
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

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        x = self.shared_conv(x)
        
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)
        
        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        
        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
                            x, x_img, query_embeds,
                            bev_pos_embeds, rv_pos_embeds,
                            attn_masks=attn_mask
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)

        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas)
    
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
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
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
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append([labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append([bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append([bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
        
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
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

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
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
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
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split  * self.split
        
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
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
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
        all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['center'][dec_id], preds_dict[0]['height'][dec_id],
                    preds_dict[0]['dim'][dec_id], preds_dict[0]['rot'][dec_id],
                    preds_dict[0]['vel'][dec_id]),
                    dim=-1
                )
                all_pred_bboxes[dec_id].append(pred_bbox)
                all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][dec_id])
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
        
        dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
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
                dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][dec_id])
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


@HEADS.register_module()
class CmtImageHead(CmtHead):

    def __init__(self, *args, **kwargs):
        super(CmtImageHead, self). __init__(*args, **kwargs)
        self.shared_conv = None

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x is None
        ret_dicts = []
        
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(len(img_metas), reference_points, img_metas)
        
        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
                            x_img, query_embeds,
                            rv_pos_embeds,
                            attn_masks=attn_mask,
                            bs=len(img_metas)
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)

        return ret_dicts


@HEADS.register_module()
class CmtLidarHead(CmtHead):

    def __init__(self, *args, **kwargs):
        super(CmtLidarHead, self). __init__(*args, **kwargs)
        self.rv_embedding = None

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        return bev_embeds, None
    
    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x_img is None

        ret_dicts = []
        x = self.shared_conv(x)
        
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)
        
        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        bev_query_embeds, _ = self.query_embed(reference_points, img_metas)

        query_embeds = bev_query_embeds
        outs_dec, _ = self.transformer(
                            x, mask, query_embeds,
                            bev_pos_embeds,
                            attn_masks=attn_mask
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)

        return ret_dicts