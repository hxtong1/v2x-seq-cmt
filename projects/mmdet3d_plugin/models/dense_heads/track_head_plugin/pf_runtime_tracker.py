# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
from .track_instance import Instances
import torch
import numpy as np


class RunTimeTracker:
    def __init__(self, output_threshold=0.2, score_threshold=0.4, record_threshold=0.4,
                      max_age_since_update=1, iou_threshold=0.1, min_active=1,
                      num_track_instance=300, train_score_thresh=None, predict_score_thresh=None):
        self.current_id = 1

        self.threshold = score_threshold
        self.output_threshold = output_threshold
        self.record_threshold = record_threshold
        self.max_age_since_update = max_age_since_update
        self.iou_threshold = iou_threshold
        self.min_active = min_active
        self.num_track_instance = int(num_track_instance)
        self.train_score_thresh = score_threshold if train_score_thresh is None else train_score_thresh
        self.predict_score_thresh = score_threshold if predict_score_thresh is None else predict_score_thresh

    def _get_scores(self, track_instances):
        if hasattr(track_instances, "scores"):
            return track_instances.scores
        if hasattr(track_instances, "cache_scores"):
            return track_instances.cache_scores
        return track_instances.pred_logits.sigmoid().max(dim=-1).values

    def _safe_topk_mask(self, scores, base_mask, k):
        out = torch.zeros_like(base_mask, dtype=torch.bool)
        valid_num = int(base_mask.sum().item())
        if valid_num <= 0 or k <= 0:
            return out
        k = min(int(k), valid_num)
        masked_scores = scores.clone()
        fill_value = torch.finfo(masked_scores.dtype).min
        masked_scores[~base_mask] = fill_value
        _, idx = torch.topk(masked_scores, k=k, dim=0)
        out[idx] = True
        return out
    
    def update_active_tracks(self, track_instances, active_mask):
        live_mask = torch.zeros_like(track_instances.obj_idxes).bool().detach()
        for i in range(len(track_instances)):
            if active_mask[i]:
                track_instances.disappear_time[i] = 0
                live_mask[i] = True
            elif track_instances.track_query_mask[i]:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] < self.max_age_since_update:
                    live_mask[i] = True
        next_instances = track_instances[live_mask]
        if self.num_track_instance > 0 and len(next_instances) > self.num_track_instance:
            scores = self._get_scores(next_instances)
            keep_mask = self._safe_topk_mask(
                scores,
                torch.ones_like(scores, dtype=torch.bool),
                self.num_track_instance)
            next_instances = next_instances[keep_mask]
        return next_instances
    
    def get_active_mask(self, track_instances, training=True):
        scores = self._get_scores(track_instances)
        if hasattr(track_instances, "track_query_mask"):
            valid_mask = track_instances.track_query_mask.bool()
        else:
            valid_mask = torch.ones_like(scores, dtype=torch.bool)

        if training:
            active_mask = valid_mask & (track_instances.matched_gt_idxes >= 0)
            if hasattr(track_instances, "iou"):
                active_mask = active_mask & (track_instances.iou > self.iou_threshold)
            matched_flag = active_mask & (scores >= self.train_score_thresh)
            if self.num_track_instance > 0:
                selected = torch.zeros_like(valid_mask, dtype=torch.bool)
                if matched_flag.any():
                    force_num = min(int(matched_flag.sum().item()), self.num_track_instance)
                    selected |= self._safe_topk_mask(scores, matched_flag, force_num)
                remain = self.num_track_instance - int(selected.sum().item())
                if remain > 0:
                    candidate_mask = valid_mask & (~selected)
                    selected |= self._safe_topk_mask(scores, candidate_mask, remain)
                active_mask = selected
            if active_mask.sum() < self.min_active and len(track_instances) > 0:
                topk = min(self.min_active, len(track_instances))
                force_mask = self._safe_topk_mask(scores, valid_mask, topk)
                active_mask = active_mask | force_mask
        else:
            score_mask = valid_mask & (scores >= self.predict_score_thresh)
            if self.num_track_instance > 0:
                if score_mask.any():
                    active_mask = self._safe_topk_mask(scores, score_mask, self.num_track_instance)
                else:
                    active_mask = self._safe_topk_mask(scores, valid_mask, self.num_track_instance)
            else:
                active_mask = score_mask
        return active_mask

    def get_assign_ids(self, track_instances, active_mask=None):
        if active_mask is None:
            active_mask = torch.ones_like(track_instances.obj_idxes, dtype=torch.bool)
        need_new_id_mask = active_mask & (track_instances.obj_idxes < 0)
        total_new_ids = int(need_new_id_mask.sum().item())
        if total_new_ids > 0:
            new_ids = torch.arange(
                self.current_id,
                self.current_id + total_new_ids,
                device=track_instances.obj_idxes.device,
                dtype=track_instances.obj_idxes.dtype)
            self.current_id += total_new_ids
            track_instances.obj_idxes[need_new_id_mask] = new_ids
            if hasattr(track_instances, "track_query_mask"):
                track_instances.track_query_mask[need_new_id_mask] = True
        return track_instances
    
    def empty(self):
        """Copy the historical buffer parts from the init
        """
        self.current_id = 1
