import math
import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from einops import rearrange
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, ModuleList

import numpy as np

from mmcv.cnn.bricks.transformer import (
    BaseTransformerLayer,
    TransformerLayerSequence,
    build_transformer_layer_sequence
)
from mmcv.cnn import (
    build_activation_layer,
    build_conv_layer,
    build_norm_layer,
    xavier_init
)
from mmcv.cnn.bricks.registry import (
    ATTENTION,TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE
)
from mmcv.utils import (
    ConfigDict,
    build_from_cfg,
    deprecated_api_warning,
    to_2tuple
)
from mmdet.models.utils.builder import TRANSFORMER


@ATTENTION.register_module()
class PETRMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


from .attention import FlashMHA

@ATTENTION.register_module()
class PETRMultiheadFlashAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=None)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


# @TRANSFORMER_LAYER.register_module()
# class PETRTransformerEncoderLayer(BaseModule):
#     """Implements encoder layer in DETR transformer.
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
#             Configs for self_attention or cross_attention, the order
#             should be consistent with it in `operation_order`. If it is
#             a dict, it would be expand to the number of attention in
#             `operation_order`.
#     """

#     def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
#                  **kwargs):

#         super(PETRTransformerEncoderLayer, self).__init__(*args, **kwargs)
#         self.return_intermediate = return_intermediate

#         self.num_points_in_pillar = num_points_in_pillar
#         self.pc_range = pc_range
#         self.fp16_enabled = False

#     @staticmethod
#     def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
#         """Get the reference points used in SCA and TSA.
#         Args:
#             H, W: spatial shape of bev.
#             Z: hight of pillar.
#             D: sample D points uniformly from each pillar.
#             device (obj:`device`): The device where
#                 reference_points should be.
#         Returns:
#             Tensor: reference points used in decoder, has \
#                 shape (bs, num_keys, num_levels, 2).
#         """

#         # reference points in 3D space, used in spatial cross-attention (SCA)
#         if dim == '3d':
#             zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
#                                 device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
#             xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
#                                 device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
#             ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
#                                 device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
#             ref_3d = torch.stack((xs, ys, zs), -1)
#             ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
#             ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
#             return ref_3d

#         # reference points on 2D bev plane, used in temporal self-attention (TSA).
#         elif dim == '2d':
#             ref_y, ref_x = torch.meshgrid(
#                 torch.linspace(
#                     0.5, H - 0.5, H, dtype=dtype, device=device),
#                 torch.linspace(
#                     0.5, W - 0.5, W, dtype=dtype, device=device)
#             )
#             ref_y = ref_y.reshape(-1)[None] / H
#             ref_x = ref_x.reshape(-1)[None] / W
#             ref_2d = torch.stack((ref_x, ref_y), -1)
#             ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
#             return ref_2d


#     def forward(self,
#                 bev_query,
#                 key,
#                 value,
#                 *args,
#                 bev_h=None,
#                 bev_w=None,
#                 bev_pos=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 valid_ratios=None,
#                 prev_bev=None,
#                 shift=0.,
#                 img_metas=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoder`.
#         Args:
#             bev_query (Tensor): Input BEV query with shape
#                 `(num_query, bs, embed_dims)`.
#             key & value (Tensor): Input multi-cameta features with shape
#                 (num_cam, num_value, bs, embed_dims)
#             reference_points (Tensor): The reference
#                 points of offset. has shape
#                 (bs, num_query, 4) when as_two_stage,
#                 otherwise has shape ((bs, num_query, 2).
#             valid_ratios (Tensor): The radios of valid
#                 points on the feature map, has shape
#                 (bs, num_levels, 2)
#         Returns:
#             Tensor: Results with shape [1, num_query, bs, embed_dims] when
#                 return_intermediate is `False`, otherwise it has shape
#                 [num_layers, num_query, bs, embed_dims].
#         """

#         output = bev_query
#         intermediate = []

#         ref_3d = self.get_reference_points(
#             bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
#         ref_2d = self.get_reference_points(
#             bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

#         # reference_points_cam, bev_mask = self.point_sampling(
#         #     ref_3d, self.pc_range, img_metas)

#         # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
#         shift_ref_2d = ref_2d  # .clone()
#         shift_ref_2d += shift[:, None, None, :]

#         # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
#         bev_query = bev_query.permute(1, 0, 2)
#         bev_pos = bev_pos.permute(1, 0, 2)
#         bs, len_bev, num_bev_level, _ = ref_2d.shape
#         if prev_bev is not None:
#             prev_bev = prev_bev.permute(1, 0, 2)
#             prev_bev = torch.stack(
#                 [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
#             hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
#                 bs*2, len_bev, num_bev_level, 2)
#         else:
#             hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
#                 bs*2, len_bev, num_bev_level, 2)

#         for lid, layer in enumerate(self.layers):
#             output = layer(
#                 bev_query,
#                 key,
#                 value,
#                 *args,
#                 bev_pos=bev_pos,
#                 ref_2d=hybird_ref_2d,
#                 ref_3d=ref_3d,
#                 bev_h=bev_h,
#                 bev_w=bev_w,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 # reference_points_cam=reference_points_cam,
#                 # bev_mask=bev_mask,
#                 prev_bev=prev_bev,
#                 **kwargs)

#             bev_query = output
#             if self.return_intermediate:
#                 intermediate.append(output)

#         if self.return_intermediate:
#             return torch.stack(intermediate)

#         return output


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


@TRANSFORMER_LAYER.register_module()
class PETRTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 with_cp=True,
                 **kwargs):
        super(PETRTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_checkpoint = with_cp
    
    def _forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerDecoderLayer, self).forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                )

        return x

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask
            )
        
        return x
