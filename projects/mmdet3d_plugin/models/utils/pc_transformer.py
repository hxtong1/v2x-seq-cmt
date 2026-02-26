import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch import Tensor
from torchvision.transforms.functional import rotate
from torch.nn.init import normal_
from mmcv.cnn import xavier_init
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner.base_module import BaseModule
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS
from mmdet.models.utils.builder import TRANSFORMER
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from mmdet.core import multi_apply
from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING
from mmcv.runner import force_fp32, auto_fp16

class PixelWeightedFusion(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusion, self).__init__()
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        return x_1


@POSITIONAL_ENCODING.register_module()
class PositionEmbeddingLearned(nn.Module):
    """Position embedding with learnable parameters.
    """
    def __init__(self, input_channels, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channels, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),)

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embediding =  self.position_embedding_head(xyz)
        return position_embediding


@POSITIONAL_ENCODING.register_module()
class ZeroPositionalEncoding(nn.Module):
    """No-op positional encoding for RoPE-only experiments."""

    def __init__(self, num_feats=128):
        super().__init__()
        self.num_feats = num_feats

    def forward(self, mask):
        # Match output shape expected by BEVFormer positional encoding:
        # [bs, 2*num_feats, h, w]
        bs, h, w = mask.shape
        return mask.new_zeros((bs, self.num_feats * 2, h, w), dtype=torch.float32)

class RoPE(nn.Module):
  def __init__(self, d: int, base: int = 10_000):
    super().__init__()
    self.base = base
    self.d = d
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):
    if self.cos_cached is not None \
        and x.shape[0] <= self.cos_cached.shape[0]:
      return
    seq_len = x.shape[0]
    theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]
    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]
    self.cos_cached = idx_theta2.cos()[:, None, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
    self.sin_cached = idx_theta2.sin()[:, None, None, :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

  def _neg_half(self, x: torch.Tensor):
    d_2 = self.d // 2 #
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

  def forward(self, x: torch.Tensor):
    self._build_cache(x)
    neg_half_x = self._neg_half(x)
    x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
    return x_rope

@TRANSFORMER.register_module()
class LIDARPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 input_channel=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                #  use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(LIDARPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.input_channel = input_channel
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        # self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        # self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(self.input_channel, self.embed_dims, kernel_size=1),
                    nn.BatchNorm2d(self.embed_dims),
                )
            )
        # self.cams_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        # normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            img_metas=None):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in img_metas])
        delta_y = np.array([each['can_bus'][1]
                           for each in img_metas])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in img_metas])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = img_metas[i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in img_metas])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            feat = self.input_proj[lvl](feat)
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat = feat.flatten(2).permute(0,2,1)
            # feat = feat + self.level_embeds[lvl:lvl + 1, :].view(1,1,-1).to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            # feat_flatten.append(feat)
            feat_ = feat.flatten(2)  # [B, C, H*W]
            feat_ = feat_.permute(0, 2, 1)  # [B, H*W, C]
            # 加 level embedding
            lvl_embed = self.level_embeds[lvl].view(1, 1, c)  # [1, 1, 256]
            feat_ = feat_ + lvl_embed  # [B, H*W, C]
            feat_flatten.append(feat_)

        # feat_flatten = torch.cat(feat_flatten, 2)
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            img_metas=img_metas,
        )

        return bev_embed
    
    def get_states_and_refs(
        self,
        bev_embed,
        query_feats,
        query_embeds,
        bev_h,
        bev_w,
        reference_points,
        reg_branches=None,
        cls_branches=None,
        img_metas=None
    ):
        query_pos = query_embeds
        query = query_feats
        reference_points = reference_points

        init_reference_out = reference_points
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=None,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            img_metas=img_metas
        )
        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out