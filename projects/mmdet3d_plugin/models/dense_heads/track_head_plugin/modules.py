import torch
import torch.nn.functional as F
from torch import nn
from .track_instance import Instances


class VelocityCalculator(nn.Module):
    """Post-process velocity predictions for stable output."""

    def __init__(self, ignore_initial=False, invalid_motion_cls=None):
        super().__init__()
        self.ignore_initial = bool(ignore_initial)
        self.invalid_motion_cls = list(invalid_motion_cls or [])

    def process(self, vels, labels=None, is_init_frame=False):
        if vels.numel() == 0:
            return vels
        out_vels = vels
        if self.ignore_initial and is_init_frame:
            out_vels = torch.zeros_like(out_vels)

        if labels is not None and len(self.invalid_motion_cls) > 0:
            invalid_cls = labels.new_tensor(self.invalid_motion_cls).view(1, -1)
            valid = (labels.view(-1, 1) != invalid_cls).all(dim=1)
            out_vels = out_vels * valid.to(out_vels.dtype).unsqueeze(-1)
        return out_vels

# MemoryBank
class MemoryBank(nn.Module):

    def __init__(self,
                 args,
                 dim_in, hidden_dim, dim_out,
                 ):
        super().__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3
        self.max_his_length = args['memory_bank_len']
        self.confidence_decay = args.get('memory_bank_confidence_decay', 0.6)

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def _ensure_memory_fields(self, track_instances, embed):
        """Initialize optional memory fields for backward compatibility."""
        n = len(track_instances)
        device = embed.device
        if (not hasattr(track_instances, "mem_bank")
                or track_instances.mem_bank.shape[:2] != (n, self.max_his_length)):
            track_instances.mem_bank = torch.zeros(
                (n, self.max_his_length, embed.shape[-1]),
                dtype=embed.dtype, device=device)
        if (not hasattr(track_instances, "mem_padding_mask")
                or track_instances.mem_padding_mask.shape[:2] != (n, self.max_his_length)):
            track_instances.mem_padding_mask = torch.ones(
                (n, self.max_his_length), dtype=torch.bool, device=device)
        if (not hasattr(track_instances, "mem_bank_score")
                or track_instances.mem_bank_score.shape[:2] != (n, self.max_his_length)):
            track_instances.mem_bank_score = torch.zeros(
                (n, self.max_his_length), dtype=embed.dtype, device=device)
        if hasattr(track_instances, "pred_boxes"):
            box_dim = track_instances.pred_boxes.shape[-1]
            if (not hasattr(track_instances, "mem_bank_anchors")
                    or track_instances.mem_bank_anchors.shape[:3] != (n, self.max_his_length, box_dim)):
                track_instances.mem_bank_anchors = torch.zeros(
                    (n, self.max_his_length, box_dim),
                    dtype=track_instances.pred_boxes.dtype,
                    device=track_instances.pred_boxes.device)
        if not hasattr(track_instances, "save_period"):
            track_instances.save_period = torch.zeros((n,), dtype=embed.dtype, device=device)

    def update(self, track_instances):
        embed = track_instances.output_embedding[:, None]  #( N, 1, 256)
        scores = track_instances.scores
        self._ensure_memory_fields(track_instances, embed[:, 0])
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
            save_embed = self.save_proj(saved_embed)  # [M, 1, C]
            prev_embed = track_instances.mem_bank[saved_idxes]  # [M, L, C]
            prev_scores = track_instances.mem_bank_score[saved_idxes] * float(self.confidence_decay)  # [M, L]
            curr_scores = scores[saved_idxes].unsqueeze(-1).to(prev_scores.dtype)  # [M, 1]
            cand_embed = torch.cat([prev_embed, save_embed], dim=1)  # [M, L+1, C]
            cand_scores = torch.cat([prev_scores, curr_scores], dim=1)  # [M, L+1]

            topk_scores, topk_idx = torch.topk(cand_scores, k=self.max_his_length, dim=1)
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, cand_embed.size(-1))
            selected_embed = torch.gather(cand_embed, 1, gather_idx)
            selected_mask = topk_scores <= 0

            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank_score = track_instances.mem_bank_score.clone()
            track_instances.mem_padding_mask = track_instances.mem_padding_mask.clone()
            track_instances.mem_bank[saved_idxes] = selected_embed
            track_instances.mem_bank_score[saved_idxes] = topk_scores
            track_instances.mem_padding_mask[saved_idxes] = selected_mask

            # Optional anchor queue aligned with temporal bank.
            if hasattr(track_instances, "mem_bank_anchors") and hasattr(track_instances, "pred_boxes"):
                prev_anchors = track_instances.mem_bank_anchors[saved_idxes]  # [M, L, B]
                curr_anchors = track_instances.pred_boxes[saved_idxes][:, None, :]  # [M, 1, B]
                cand_anchors = torch.cat([prev_anchors, curr_anchors], dim=1)  # [M, L+1, B]
                gather_anchor_idx = topk_idx.unsqueeze(-1).expand(-1, -1, cand_anchors.size(-1))
                selected_anchors = torch.gather(cand_anchors, 1, gather_anchor_idx)
                track_instances.mem_bank_anchors = track_instances.mem_bank_anchors.clone()
                track_instances.mem_bank_anchors[saved_idxes] = selected_anchors

    def _forward_temporal_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]

        # Any valid historical slot can enable temporal refinement.
        valid_idxes = (~key_padding_mask).any(dim=1)
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
