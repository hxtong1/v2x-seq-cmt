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

from functools import reduce
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
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
            ret_dict[head] = rearrange(
                head_output, "b (n c) q -> n b q c", n=N)

        return ret_dict


@HEADS.register_module()
class CmtHead(BaseModule):

    def __init__(self,
                 in_channels,
                 num_query=900,
                 hidden_dim=128,
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
                 tasks=[
                     dict(num_class=3, class_names=[
                          'car', 'bicycle', 'pedestrian'])
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                     type="L1Loss",
                     reduction="mean",
                     loss_weight=0.25,
                 ),
                 loss_iou=None,
                 loss_heatmap=dict(
                     type="GaussianFocalLoss",
                     reduction="mean"
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 ref_point_x_range=None,
                 **kwargs):
        assert init_cfg is None
        super(CmtHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        # ref_point_x_range: [low, high] for x (norm) to bias queries to far range (infra)
        self.ref_point_x_range = ref_point_x_range

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou) if loss_iou is not None else None
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False

        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )

        # transformer
        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def init_weights(self):
        super(CmtHead, self).init_weights()
        w = self.reference_points.weight.data  # (num_query, 3)
        nn.init.uniform_(w, 0, 1)
        if self.ref_point_x_range is not None:
            low, high = self.ref_point_x_range
            nn.init.uniform_(w[:, 0], low, high)  # bias x (first dim) to far range

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0)  # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = []
            for img_meta in img_metas:
                gt_bboxes = img_meta.get('gt_bboxes_3d', None)
                if isinstance(gt_bboxes, list):
                    # Take the first item if it's a list (caused by batch dimension changes)
                    gt_bboxes = gt_bboxes[0]
                if hasattr(gt_bboxes, '_data'):
                    gt_bboxes = gt_bboxes._data

                if gt_bboxes is not None:
                    targets.append(
                        torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1))
                else:
                    targets.append(torch.zeros(
                        (0, 9), device=reference_points.device))  # fallback

            labels = []
            for img_meta in img_metas:
                gt_labels = img_meta.get('gt_labels_3d', None)
                if isinstance(gt_labels, list):
                    gt_labels = gt_labels[0]
                if hasattr(gt_labels, '_data'):
                    gt_labels = gt_labels._data
                if gt_labels is not None:
                    labels.append(gt_labels)
                else:
                    labels.append(torch.zeros((0,), dtype=torch.long,
                                  device=reference_points.device))  # fallback

            known = [(torch.ones_like(t)).to(reference_points.device)
                     for t in labels]
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
            if max(known_num) == 0:
                groups = self.scalar
            else:
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

            single_pad = max(1, int(max(known_num))
                             ) if self.training else int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat(
                [padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat(
                    [torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(
                reference_points.device) < 0
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
            padded_reference_points = reference_points.unsqueeze(
                0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(
            H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(
            W, device=img_feats[0].device).float() * pad_w / W
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float(
        ) * (self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid(
            [coords_h, coords_w, coords_d])

        coords = torch.stack(
            [coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        imgs2lidars = np.concatenate(
            [np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :])\
            / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None, None, :]
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(
            pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(
            lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img'])
                               for meta in img_metas])
        imgs2lidars = torch.from_numpy(
            imgs2lidars).float().to(ref_points.device)

        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(
            self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        proj_points = torch.einsum('bnd, bvcd -> bvnc', torch.cat(
            [ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1), lidars2imgs)

        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / \
            (proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape)

        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (
            proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (
            self.pc_range[3] - 1) / self.depth_num
        proj_points_clone = torch.einsum(
            'bvnc, d -> bvndc', proj_points_clone, coords_d)
        proj_points_clone = torch.cat([proj_points_clone[..., :3], proj_points_clone.new_ones(
            *proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum(
            'bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None, :])\
            / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(self.pc_range[:3]))[None, None, None, :]

        rv_embeds = self.rv_embedding(
            projback_points.reshape(*projback_points.shape[:-2], -1))
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        rv_embeds = self._rv_query_embed(ref_points, img_metas)
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        conv_dtype = self.shared_conv.conv.weight.dtype
        if x.dtype != conv_dtype:
            x = x.to(dtype=conv_dtype)
        x = self.shared_conv(x)

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            x.shape[0], reference_points, img_metas)

        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])

        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(
            pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        bev_query_embeds, rv_query_embeds = self.query_embed(
            reference_points, img_metas)
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
            _center, _height = center.new_zeros(
                center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * \
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * \
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * \
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            outs['norm_center'] = center
            outs['norm_height'] = height

            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

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
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (
                    new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:,
                                                  :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict

            ret_dicts.append(outs)

        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None, track_instances=None):
        """
            list([bs, c, h, w])
        """
        img_metas_list = [img_metas for _ in range(len(pts_feats))]
        track_instances_list = [track_instances for _ in range(len(pts_feats))]
        if img_feats is None:
            img_feats_list = [None for _ in range(len(pts_feats))]
        else:
            img_feats_list = img_feats
        return multi_apply(self.forward_single, pts_feats, img_feats_list, img_metas_list, track_instances_list)

    def _split_gt_by_task(self, gt_bboxes_3d, gt_labels_3d, gt_inds=None):
        """Split GT boxes/labels/instance ids by task for assigner use."""
        if isinstance(gt_labels_3d, list):
            device = gt_labels_3d[0].device
            gt_labels_3d = gt_labels_3d[0]
        else:
            device = gt_labels_3d.device

        if isinstance(gt_bboxes_3d, list):
            gt_bboxes_3d = gt_bboxes_3d[0]
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)

        if gt_inds is not None:
            if isinstance(gt_inds, list):
                gt_inds = gt_inds[0]
            gt_inds = gt_inds.to(device=device, dtype=torch.long).reshape(-1)

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
        task_gt_indices = []
        task_obj_ids = []
        flag2 = 0
        for mask in task_masks:
            task_box = []
            task_class = []
            task_gt_index = []
            task_obj_id = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
                task_gt_index.append(m[0].to(device=device, dtype=torch.long))
                if gt_inds is not None:
                    task_obj_id.append(gt_inds[m].to(
                        device=device, dtype=torch.long))
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            task_gt_indices.append(torch.cat(task_gt_index).long().to(device))
            if gt_inds is not None:
                task_obj_ids.append(torch.cat(task_obj_id).long().to(device))
            else:
                task_obj_ids.append(None)
            flag2 += len(mask)

        return task_boxes, task_classes, task_gt_indices, task_obj_ids

    @torch.no_grad()
    def assign_track_targets(self,
                             gt_bboxes_3d,
                             gt_labels_3d,
                             pred_bboxes,
                             pred_logits,
                             gt_inds=None):
        """Match current frame queries to GT and expose GT/object ids for tracker."""
        if not isinstance(pred_bboxes, (list, tuple)):
            pred_bboxes = [pred_bboxes]
        if not isinstance(pred_logits, (list, tuple)):
            pred_logits = [pred_logits]

        task_boxes, task_classes, task_gt_indices, task_obj_ids = \
            self._split_gt_by_task(gt_bboxes_3d, gt_labels_3d, gt_inds=gt_inds)

        device = pred_bboxes[0].device
        per_query_matches = {}

        for task_id, (bbox_pred, logits_pred, gt_bboxes, gt_labels, gt_index_map, obj_id_map) in enumerate(
            zip(pred_bboxes, pred_logits, task_boxes,
                task_classes, task_gt_indices, task_obj_ids)
        ):
            if bbox_pred.numel() == 0 or logits_pred.numel() == 0 or gt_bboxes.numel() == 0:
                continue

            bbox_pred = bbox_pred.float()
            logits_pred = logits_pred.float()
            assign_results = self.assigner.assign(
                bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(
                assign_results, bbox_pred, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            pos_gt_inds = sampling_result.pos_assigned_gt_inds
            if pos_inds.numel() == 0:
                continue

            pred_scores = logits_pred.sigmoid().max(dim=-1).values
            for query_idx, local_gt_idx in zip(pos_inds.tolist(), pos_gt_inds.tolist()):
                score = pred_scores[query_idx]
                prev_match = per_query_matches.get(query_idx)
                if prev_match is not None and prev_match["track_scores"] >= score:
                    continue
                per_query_matches[query_idx] = dict(
                    matched_gt_idxes=gt_index_map[local_gt_idx],
                    obj_idxes=obj_id_map[local_gt_idx] if obj_id_map is not None else gt_index_map.new_tensor(
                        -1),
                    labels=gt_labels[local_gt_idx],
                    track_scores=score,
                    task_id=task_id,
                )

        if len(per_query_matches) == 0:
            empty_long = torch.zeros(0, device=device, dtype=torch.long)
            empty_float = torch.zeros(
                0, device=device, dtype=pred_bboxes[0].dtype)
            return dict(
                query_inds=empty_long,
                matched_gt_idxes=empty_long,
                obj_idxes=empty_long,
                labels=empty_long,
                track_scores=empty_float,
            )

        query_inds = torch.tensor(
            sorted(per_query_matches.keys()), device=device, dtype=torch.long)
        matched_gt_idxes = torch.stack(
            [per_query_matches[idx]["matched_gt_idxes"]
                for idx in query_inds.tolist()]
        ).long()
        obj_idxes = torch.stack(
            [per_query_matches[idx]["obj_idxes"]
                for idx in query_inds.tolist()]
        ).long()
        labels = torch.stack(
            [per_query_matches[idx]["labels"] for idx in query_inds.tolist()]
        ).long()
        track_scores = torch.stack(
            [per_query_matches[idx]["track_scores"]
                for idx in query_inds.tolist()]
        )
        return dict(
            query_inds=query_inds,
            matched_gt_idxes=matched_gt_idxes,
            obj_idxes=obj_idxes,
            labels=labels,
            track_scores=track_scores,
        )

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
        task_boxes, task_classes, _, _ = self._split_gt_by_task(
            gt_bboxes_3d, gt_labels_3d, gt_inds=None)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]

            # Clean up abnormal GT velocity for the assigner so it doesn't break matching
            if gt_bboxes.size(-1) > 7:
                vel_norm = torch.norm(gt_bboxes[..., 7:9], dim=-1)
                abnormal_mask = vel_norm > 30.0
                if abnormal_mask.any():
                    gt_bboxes = gt_bboxes.clone()
                    gt_bboxes[abnormal_mask, 7:9] = 30.0

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

        # mask out abnormal velocity
        if bbox_targets.size(-1) > 7:
            vel_norm = torch.norm(bbox_targets[..., 7:9], dim=-1)
            abnormal_vel_mask = vel_norm > 30.0  # mask if speed > 30m/s
            bbox_weights[abnormal_vel_mask, 8:10] = 0.0

        # P0: reorder pred to match target layout
        pred_for_loss = pred_bboxes_flatten[isnotnan, :10][:, [
            0, 1, 3, 4, 2, 5, 6, 7, 8, 9]]
        target_for_loss = normalized_bbox_targets[isnotnan, :10].clone()

        # Center in physical space (no P1 norm) so 1m error → strong gradient for ATE
        # P1 reverted: center [0,1] made gradient too weak, ATE stayed high

        loss_bbox = self.loss_bbox(
            pred_for_loss,
            target_for_loss,
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_iou = pred_for_loss.sum() * 0.0
        if self.loss_iou is not None and num_total_pos > 0 and isnotnan.any():
            loss_iou = self.loss_iou(
                pred_for_loss,
                target_for_loss,
                weight=None,
                avg_factor=num_total_pos,
                pc_range=list(self.pc_range),
            )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_iou = torch.nan_to_num(loss_iou)
        return loss_cls, loss_bbox, loss_iou

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
        loss_cls_tasks, loss_bbox_tasks, loss_iou_tasks = multi_apply(
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
        num_pos_batch = sum(num_total_pos)
        loss_iou = sum(loss_iou_tasks) if loss_iou_tasks else pred_bboxes[0].sum() * 0.0
        return sum(loss_cls_tasks), sum(loss_bbox_tasks), loss_iou, num_pos_batch

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

        if num_tgt == 0:
            return pred_logits.sum() * 0.0, pred_bboxes.sum() * 0.0

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
        # To prevent distributed desync when has_dn is True on some GPUs but False on others,
        # we do not use reduce_mean here.
        num_tgt = torch.clamp(num_tgt, min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * \
            bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        loss_bbox = self.loss_bbox(
            pred_bboxes, normalized_bbox_targets, bbox_weights[isnotnan, :10], avg_factor=num_tgt)

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
        # We need to make dn_mask_dict accessible by task
        # Currently _dn_loss_single_task expects a single mask_dict, but multi_apply will pass lists.
        # Ensure dn_mask_dict is formatted correctly for multi_apply.
        # But wait, dn_mask_dict inside CmtHead represents the whole batch and all tasks together in the original logic?
        # Actually in CmtHead it's created per task. Let's wrap it in a list if it's not.
        if not isinstance(dn_mask_dict, list):
            dn_mask_dict = [dn_mask_dict] * len(pred_bboxes)
        elif len(dn_mask_dict) == 1 and len(pred_bboxes) > 1:
            dn_mask_dict = dn_mask_dict * len(pred_bboxes)

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
                # We need to compute loss only for newborn queries, not track queries.
                # Usually we can get newborn predictions by slicing from -num_query:
                # But here to be safe and simple, let's assume we pass all predictions
                # (track + newborn) to HungarianAssigner. We'll handle this detail later
                # in the tracker if needed, or slice here if num_query is known.
                # The prompt requested focusing on CMTHead adapter first.
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

        loss_cls, loss_bbox, loss_iou, num_pos_list = multi_apply(
            self.loss_single, all_pred_bboxes, all_pred_logits,
            [gt_bboxes_3d for _ in range(num_decoder)],
            [gt_labels_3d for _ in range(num_decoder)],
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_bbox'] = loss_bbox[-1]
        loss_dict['loss_iou'] = loss_iou[-1]
        # num_pos: Hungarian-matched positives per batch (last decoder), for IoU3D feasibility check
        loss_dict['num_pos'] = loss_cls[-1].new_tensor(
            float(num_pos_list[-1]), dtype=torch.float32)

        aux_loss_cls = loss_cls[-1] * 0.0
        aux_loss_bbox = loss_bbox[-1] * 0.0
        aux_loss_iou = loss_iou[-1] * 0.0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(loss_cls[:-1],
                                                       loss_bbox[:-1],
                                                       loss_iou[:-1]):
            aux_loss_cls = aux_loss_cls + loss_cls_i
            aux_loss_bbox = aux_loss_bbox + loss_bbox_i
            aux_loss_iou = aux_loss_iou + loss_iou_i
        loss_dict['aux_loss_cls'] = aux_loss_cls
        loss_dict['aux_loss_bbox'] = aux_loss_bbox
        loss_dict['aux_loss_iou'] = aux_loss_iou

        dn_pred_bboxes, dn_pred_logits = collections.defaultdict(
            list), collections.defaultdict(list)
        dn_mask_dicts = collections.defaultdict(list)
        has_dn = False
        for task_id, preds_dict in enumerate(preds_dicts, 0):
            if 'dn_center' in preds_dict[0]:
                has_dn = True
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

        if has_dn:
            dn_pred_bboxes = [dn_pred_bboxes[idx]
                              for idx in range(num_decoder)]
            dn_pred_logits = [dn_pred_logits[idx]
                              for idx in range(num_decoder)]
            dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
            dn_loss_cls, dn_loss_bbox = multi_apply(
                self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
            )

            loss_dict['dn_loss_cls'] = dn_loss_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_loss_bbox[-1]

            # Per-layer dn_loss (same total as dn_aux; no double-count)
            for num_dec_layer, (loss_cls_i, loss_bbox_i) in enumerate(
                    zip(dn_loss_cls[:-1], dn_loss_bbox[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
        else:
            dummy_loss = loss_dict['loss_cls'] * 0.0
            loss_dict['dn_loss_cls'] = dummy_loss
            loss_dict['dn_loss_bbox'] = dummy_loss
            loss_dict['dn_aux_loss_cls'] = dummy_loss
            loss_dict['dn_aux_loss_bbox'] = dummy_loss

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
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            len(img_metas), reference_points, img_metas)

        rv_pos_embeds = self._rv_pe(x_img, img_metas)

        bev_query_embeds, rv_query_embeds = self.query_embed(
            reference_points, img_metas)
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
            _center, _height = center.new_zeros(
                center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * \
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * \
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * \
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            outs['norm_center'] = center
            outs['norm_height'] = height

            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

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
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (
                    new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:,
                                                  :, :mask_dict['pad_size'], :]
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

    def forward_single(self, x, x_img, img_metas, track_instances=None):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x_img is None

        ret_dicts = []
        conv_dtype = self.shared_conv.conv.weight.dtype
        if x.dtype != conv_dtype:
            x = x.to(dtype=conv_dtype)
        x = self.shared_conv(x)

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            x.shape[0], reference_points, img_metas)

        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])

        bev_pos_embeds = self.bev_embedding(
            pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        bev_query_embeds, _ = self.query_embed(reference_points, img_metas)

        if track_instances is not None and len(track_instances) > 0:
            if isinstance(track_instances, list):
                ref_pts_track_bs = torch.stack(
                    [inst.ref_pts for inst in track_instances], dim=0)
                query_embeds_track_bs = torch.stack(
                    [inst.query_embeds for inst in track_instances], dim=0)
                pad_size = 0
            else:
                ref_pts_track = track_instances.ref_pts
                query_embeds_track = track_instances.query_embeds
                bs = x.shape[0]
                ref_pts_track_bs = ref_pts_track.unsqueeze(0).repeat(bs, 1, 1)
                query_embeds_track_bs = query_embeds_track.unsqueeze(
                    0).repeat(bs, 1, 1)

            if mask_dict and mask_dict['pad_size'] > 0:
                pad_size = mask_dict['pad_size']
                # Keep the padding at the beginning, replace the rest with track_instances
                reference_points = torch.cat(
                    [reference_points[:, :pad_size, :], ref_pts_track_bs], dim=1)
                bev_query_embeds = torch.cat(
                    [bev_query_embeds[:, :pad_size, :], query_embeds_track_bs], dim=1)
            else:
                reference_points = ref_pts_track_bs
                bev_query_embeds = query_embeds_track_bs

            # attn_mask already has the correct shape [pad_size + num_query, pad_size + num_query]
            # Since track_instances has length num_query (900), the shape is preserved.

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
            _center, _height = center.new_zeros(
                center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * \
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * \
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * \
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            outs['norm_center'] = center
            outs['norm_height'] = height
            outs['norm_center'] = center
            outs['norm_height'] = height
            outs['query_feats'] = outs_dec

            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

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
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (
                    new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:,
                                                  :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict

            ret_dicts.append(outs)

        return ret_dicts
