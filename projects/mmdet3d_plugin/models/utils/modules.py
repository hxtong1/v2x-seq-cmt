import torch
import torch.nn.functional as F
from torch import nn
from .instance import Instances
import json
from typing import Dict
from torch import nn, Tensor
from mmdet3d.registry import MODELS
from mmcv.cnn import Scale

from enum import IntEnum

class BBox3DIndex(IntEnum):
    X = 0
    Y = 1
    Z = 2
    L = 3
    W = 4
    H = 5
    SIN_YAW = 6
    COS_YWA = 7

    VX = 8
    VY = 9
@MODELS.register_module()
class InstanceBank(nn.Module):

    def __init__(self,
                 args,
                 num_queries: int,
                 embed_dims: int,
                 query_dims: int,
                 query_grad: bool,
                 instance_grad: bool,
                 dim_in, 
                 hidden_dim, 
                 dim_out,
                 ):
        super().__init__()
        self._build_layers(args, num_queries, embed_dims, query_dims, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def init_query(self, query_json_file: str, query_grad: bool=False):
        if query_json_file is not None:
            queries = json.load(open(query_json_file, 'r'))['queries']
            queries = torch.tensor(queries)
            self.num_queries = min(self.num_queries, len(queries))
            queries = queries[:self.num_queries]
            self.queries_init = queries
            self.queries = nn.Parameter(
                torch.tensor(queries, dtype=torch.float32),
                requires_grad=query_grad
            )
            assert self.queries.shape[1] == self.query_dims
        else:
            self.queries = nn.Parameter(
                torch.zeros(self.num_queries, self.query_dims, dtype=torch.float32),
                requires_grad=query_grad
            )
            self.query_init = torch.nn.init.kaiming_normal(self.queries)
    
    def _init_instance_feats(self, instance_grad: bool):
        self.instance_feats = nn.Parameter(
            torch.zeros(self.num_queries, self.embed_dims, dtype=torch.float32),
            requeries_grad=instance_grad
        )
    def init_weights(self):
        if self.num_queries > 0:
            self.queries.data = self.queries.data.new_tensor(self.query_init)
            if self.instance_feats.requires_grad:
                torch.nn.init.xavier_uniform_(self.instance_feats.data, gain=1)
    
    def get(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        if self.num_queries == 0:
            return dict(
                laq_instance_feats=torch.zeros(batch_size, 0, self.embed_dims, devices=device),
                laq_queries=torch.zeros(batch_size, 0, self.num_queries, devices=device),
                num_laq=0,
            )
        instance_feats = self.instance_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)
        return dict(
            laq_instance_feat=instance_feats,
            laq_queries=queries,
            num_laq=self.num_queries,
        )

@MODELS.register_module()
class TemporalInstanceBank(InstanceBank):
    def __init__(self,
                *args,
                num_temp_instances: int,
                query_encoder: Dict,
                confidence_decay: float = 0.6,
                use_temp_attn_mask: bool = False,
                use_last2curr_embedding: bool = False,
                temp_attn_mask_scale = 10.0,
                with_motion: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        assert num_temp_instances > 0, "num_temp_instances must be greater than 0"
        self.num_temp_instances = num_temp_instances
        self.confidence_decay = confidence_decay
        self.use_temp_attn_mask = use_temp_attn_mask
        if self.use_temp_attn_mask:
            self.temp_attn_mask_scale = (
                Scale(temp_attn_mask_scale)
                if temp_attn_mask_scale is not None
                else None
            )

        self.temp_info = None
        self.test_temp_info = None
        
        self.query_encoder = MODELS.build(query_encoder)
        self.use_last2curr_embedding = use_last2curr_embedding
        if self.use_last2curr_embedding:
            self.last2curr_embedding = nn.Sequential(
                nn.Linear(16, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
            )
        
        self.with_motion = with_motion
    
    def train(self, mode = True):
        super().train(mode)
        if mode:
            self.test_temp_info = None
    
    def _generate_init_temp_info(self, bs: int, num_instances: int, device: torch.device) -> Dict[str, Tensor]:
        """Generate initial temp_info dictionary.
        
        Args:
            bs: Batch size
            num_instances: Number of temporal instances
            device: Device to create tensors on
            
        Returns:
            Dictionary containing initialized temp_info
        """
        seq_ids = torch.full((bs,), -1, dtype=torch.long, device=device)
        timestamps = torch.zeros((bs,), dtype=torch.float64, device=device)
        lidar2maps = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1, -1)
        confidences = torch.zeros((bs, num_instances), dtype=torch.float32, device=device)
        instance_feats = torch.zeros((bs, num_instances, self.embed_dims), dtype=torch.float32, device=device)
        queries = torch.zeros((bs, num_instances, self.query_dims), dtype=torch.float32, device=device)
        is_init = torch.zeros((bs,), dtype=torch.bool, device=device)
        last2currs = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1, -1)
        
        return dict(seq_ids=seq_ids,
                    timestamps=timestamps,
                    lidar2maps=lidar2maps,
                    is_init=is_init,
                    last2currs=last2currs,
                    confidences=confidences,
                    instance_feats=instance_feats,
                    queries=queries)
    
    @classmethod
    def get_tgt_frame_yaw(cls, src_sin_yaw: Tensor, src_cos_yaw: Tensor, Rs: Tensor) -> Tensor:
        """
        Args:
            angles: [bs, num_temp_instances, 3], last dim is (yaw, pitch, roll)
            Rs: [bs, 3, 3], rotation matrix
        Returns:
            tgt_angle: [bs, num_temp_instances, 3]
        """
        r00, r01, r10, r11 = Rs[..., 0, 0], Rs[..., 0, 1], Rs[..., 1, 0], Rs[..., 1, 1]
        tgt_cos_yaw = r00.view(-1, 1, 1) * src_cos_yaw + r01.view(-1, 1, 1) * src_sin_yaw
        tgt_sin_yaw = r10.view(-1, 1, 1) * src_cos_yaw + r11.view(-1, 1, 1) * src_sin_yaw
        
        return tgt_sin_yaw, tgt_cos_yaw
    
    def get_temp_info(self, bs: int, device: torch.device) -> Dict[str, Tensor]:
        if self.training:
            if self.temp_info is None:
                self.temp_info = self._generate_init_temp_info(bs, self.num_temp_instances, device)
            return self.temp_info
        else:
            if self.test_temp_info is None:
                self.test_temp_info = self._generate_init_temp_info(bs, self.num_temp_instances, device)
            return self.test_temp_info
    
    def update_queries_onnx(self, temp_queries: Tensor, metainfo: Dict):
        last2curr = metainfo['last2curr']
        last_position = temp_queries[..., [X, Y, Z]]
        curr_position = last_position @ last2curr[:, :3, :3].transpose(1, 2) + \
            last2curr[:, :3, 3].unsqueeze(1)
        if self.with_motion:
              last_velocities = torch.cat([temp_queries[..., [VX, VY]], torch.zeros_like(temp_queries[..., [0]])], dim=-1)
              curr_velocities = last_velocities @ last2curr[:, :3, :3].transpose(1, 2)  # [bs, num_temp_instances, 3]
              curr_position = curr_position + curr_velocities * metainfo['frame_time_gap'].unsqueeze(1)
        src_sin_yaw = temp_queries[..., [SIN_YAW]]
        src_cos_yaw = temp_queries[..., [COS_YAW]]
        curr_sin_yaw, curr_cos_yaw = self.get_tgt_frame_yaw(src_sin_yaw, src_cos_yaw, last2curr[:, :3, :3])
        temp_queries = [curr_position, temp_queries[..., [L, W, H]], curr_sin_yaw, curr_cos_yaw]
        if self.with_motion:
            temp_queries.append(curr_velocities[..., :2])

        curr_queries = torch.cat(temp_queries, dim=-1)
        return curr_queries

    def update_queries(self, temp_queries : Tensor, temp_info : Dict, metainfo: Dict):
        last2currs = temp_info['last2currs']
        is_init = temp_info['is_init']
        last_position = temp_queries[..., [X, Y, Z]]  # [bs, num_temp_instances, 3]
        curr_position = last_position @ last2currs[:, :3, :3].transpose(1, 2) + last2currs[:, :3, 3].unsqueeze(1)  # [bs, num_temp_instances, 3]
        if self.with_motion:
            last_velocities = torch.cat([temp_queries[..., -2:], torch.zeros_like(temp_queries[..., [0]])], dim=-1)
            curr_velocities = last_velocities @ last2currs[:, :3, :3].transpose(1, 2)  # [bs, num_temp_instances, 3]
            time_gap = (metainfo['timestamps'] - temp_info['timestamps']).to(curr_velocities.dtype)  # [bs], seconds
            time_gap = torch.where(is_init, torch.zeros_like(time_gap), time_gap)
            bs = len(metainfo['sequence_ids'])
            curr_position = curr_position + curr_velocities * time_gap.view(bs, 1, 1)
        src_sin_yaw = temp_queries[..., [SIN_YAW]]
        src_cos_yaw = temp_queries[..., [COS_YAW]]
        curr_sin_yaw, curr_cos_yaw = self.get_tgt_frame_yaw(src_sin_yaw, src_cos_yaw, last2currs[:, :3, :3])    
        temp_queries = [curr_position, temp_queries[..., [L, W, H]], curr_sin_yaw, curr_cos_yaw]
        if self.with_motion:
            temp_queries.append(curr_velocities[..., :2])
        temp_queries = torch.cat(temp_queries, dim=-1)
        return temp_queries
    
    def get_temp_instances_onnx(
        self,
        metainfo: Dict[str, Tensor],
        instance_info: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        temp_info = metainfo['temporal_info']
        split_dims = [self.query_dims, self.embed_dims, 1]
        temp_queries, temp_instance_feats, temp_confidences = torch.split(temp_info, split_dims, dim=-1)
        temp_queries = self.update_queries_onnx(temp_queries, metainfo)
        last2curr = metainfo['last2curr']
        temp_query_embeds = self.query_encoder(pos2posemb3d(temp_queries))  # [bs, num_temp_instances, embed_dims]
        if self.use_last2curr_embedding:
            temp_query_embeds = temp_query_embeds + self.last2curr_embedding(
                last2curr.flatten(1)).unsqueeze(1)
        
        ret = dict(temp_instance_feats=temp_instance_feats,
                    temp_query_embeds=temp_query_embeds,
                    temp_queries=temp_queries)
        
        if self.use_temp_attn_mask:
            temp_confidences = temp_confidences.squeeze(-1)
            ret['temp_attn_mask'] = temp_confidences - 1
            if self.temp_attn_mask_scale is not None:
                ret['temp_attn_mask'] = self.temp_attn_mask_scale(ret['temp_attn_mask'])
            ret['temp_confidences'] = temp_confidences
        instance_info.update(ret)
        return instance_info
    
    def get_temp_instances(
        self,
        metainfo: Dict[str, Tensor],
        instance_info: Dict[str, Tensor] = dict(),
    ) -> Dict[str, Tensor]:
        if torch.onnx.is_in_onnx_export():
            return self.get_temp_instances_onnx(metainfo, instance_info)
        
        bs = len(metainfo['sequence_ids'])
        device = metainfo['lidar2maps'].device
        temp_info = self.get_temp_info(bs, device)
        
        is_init = (temp_info["seq_ids"] != metainfo["sequence_ids"]) | (
            metainfo["timestamps"] - temp_info["timestamps"] > self.time_diff_threshold
        )
        temp_info['is_init'] = is_init

        last_lidar2maps = temp_info['lidar2maps']
        curr_map2lidars = metainfo.get('map2lidars', metainfo['lidar2maps'].inverse())
        last2currs = curr_map2lidars @ last_lidar2maps
        temp_info['last2currs'] = last2currs
        temp_queries = self.update_queries(temp_info['queries'], temp_info, metainfo)
        temp_query_embeds = self.query_encoder(pos2posemb3d(temp_queries))  # [bs, num_temp_instances, embed_dims]
        if self.use_last2curr_embedding:
            temp_query_embeds = temp_query_embeds + self.last2curr_embedding(
                last2currs.view(bs, -1)).unsqueeze(1)
        
        mask = torch.logical_and(~is_init.unsqueeze(1), temp_info['confidences'] > 0).float()  # [bs, num_temp_instances]
        temp_query_embeds = temp_query_embeds * mask.unsqueeze(-1)
        temp_instance_feats = temp_info['instance_feats']
        temp_instance_feats = temp_instance_feats * mask.unsqueeze(-1)
        
        ret = dict(temp_instance_feats=temp_instance_feats,
                   temp_query_embeds=temp_query_embeds,
                   temp_queries=temp_queries)
        
        if self.use_temp_attn_mask:
            temp_confidences = temp_info['confidences']
            temp_confidences = temp_confidences * mask
            ret["temp_attn_mask"] = temp_confidences - 1
            if self.temp_attn_mask_scale is not None:
                ret["temp_attn_mask"] = self.temp_attn_mask_scale(ret["temp_attn_mask"])

            ret['temp_confidences'] = temp_confidences
        
        temp_info['queries'] = temp_queries.detach().clone()

        instance_info.update(ret)
        return instance_info

    
    def update_temp_instances_onnx(self,
                                   instance_feats: Tensor,
                                   queries: Tensor,
                                   scores: Tensor,
                                   metainfo: Dict[str, Tensor]) -> Tensor:
        curr_info = torch.cat([queries, instance_feats, scores.unsqueeze(-1)], dim=-1)
        temp_info = metainfo['temporal_info']
        temp_info[:, :self.query_dims] = metainfo['temp_queries']
        temp_info[:, -1] = temp_info[:, -1] * self.confidence_decay
        
        output = torch.cat([temp_info, curr_info], dim=0)
        selected_inds = torch.topk(output[:, -1], self.num_temp_instances)[1]
        return output[selected_inds]
    
    def update_temp_instances(self,
                              instance_feats: Tensor,
                              queries: Tensor,
                              cls_logits: Tensor,
                              metainfo: Dict[str, Tensor]):
        if torch.onnx.is_in_onnx_export():
            return self.update_temp_instances_onnx(instance_feats, queries, cls_logits, metainfo)
        
        instance_feats = instance_feats.detach()  # [bs, num_queries, embed_dims]
        queries = queries.detach()  # [bs, num_queries, query_dims]
        confidences = cls_logits.sigmoid().max(dim=-1)[0].detach()  # [bs, num_queries]

        temp_info = self.get_temp_info(len(metainfo['sequence_ids']), instance_feats.device)
        instance_feats = torch.cat([instance_feats, temp_info['instance_feats']], dim=1)
        queries = torch.cat([queries, temp_info['queries']], dim=1)
        
        is_first = temp_info['is_init']
        temp_info['confidences'][is_first] = 0
        confidences = torch.cat([confidences, temp_info['confidences'] * self.confidence_decay], dim=1)
        
        selected_inds = torch.topk(confidences, self.num_temp_instances, dim=-1)[1]
        temp_info['queries'] = queries.gather(1, selected_inds.unsqueeze(-1).expand(-1, -1, self.query_dims))
        temp_info['instance_feats'] = instance_feats.gather(1, selected_inds.unsqueeze(-1).expand(-1, -1, self.embed_dims))
        temp_info['confidences'] = confidences.gather(1, selected_inds)
        temp_info['seq_ids'] = metainfo['sequence_ids']
        temp_info['lidar2maps'] = metainfo['lidar2maps']
        temp_info['timestamps'] = metainfo['timestamps']
        # temp_info is already a reference to self.temp_info or self.test_temp_info,
        # so no need to reassign it
      
        
    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3
        self.max_his_length = args['memory_bank_len']

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def update(self, track_instances):
        embed = track_instances.output_embedding[:, None]  #( N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            # saved_idxes = (save_period == 0)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat([mem_padding_mask[saved_idxes, 1:], torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def _forward_temporal_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]

        valid_idxes = key_padding_mask[:, -1] == 0
        embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],                  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),   # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
                key_padding_mask=key_padding_mask,
            )[0][0]

            embed = self.temporal_norm1(embed + embed2)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
            embed = self.temporal_norm2(embed + embed2)
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances


# QIM
class QueryInteractionBase(nn.Module):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()

class QueryInteractionModule(QueryInteractionBase):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args["random_drop"]
        self.fp_ratio = args["fp_ratio"]
        self.update_query_pos = args["update_query_pos"]

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args["merger_dropout"]

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args["update_query_pos"]:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        # dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        # query_pos = track_instances.query[:, :dim // 2]
        # query_feat = track_instances.query[:, dim // 2:]
        query_pos = track_instances.query_embeds
        query_feat = track_instances.query_feats
        q = k = query_pos + out_embed

        # attention
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:,
                                                                             0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            # ffn: linear_pos2
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            # track_instances.query[:, :dim // 2] = query_pos
            track_instances.query_embeds = query_pos

        query_feat2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        # track_instances.query[:, dim // 2:] = query_feat
        track_instances.query_feats = query_feat
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        # update ref_pts using track_instances.pred_boxes
        return track_instances

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
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

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        if self.training:
            active_idxes = (track_instances.obj_idxes >=
                            0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(
                active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(
                    track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[
                track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data, is_drop=False) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(
            active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        if is_drop:
            full_length = len(init_track_instances)
            active_length = len(active_track_instances)
            if active_length > 0:
                # import ipdb;ipdb.set_trace()
                random_index = torch.randperm(full_length)
                selected = random_index[:full_length-active_length]
                init_track_instances = init_track_instances[selected]
        merged_track_instances = Instances.cat(
            [init_track_instances, active_track_instances])
        return merged_track_instances