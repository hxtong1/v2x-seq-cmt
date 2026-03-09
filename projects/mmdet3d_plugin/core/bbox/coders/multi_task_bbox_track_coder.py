# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class MultiTaskBBoxTrackCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 with_nms=False,
                 iou_thres=0.3):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.with_nms = with_nms
        self.nms_iou_thres = iou_thres

    def encode(self):
        pass

    def decode_single_task(self, cls_scores, bbox_preds, task_ids):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        num_query = cls_scores.shape[0]

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()

        bbox_preds = bbox_preds[task_index * num_query + bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode_single_track(self, cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True, img_metas=None):
        """Decode bboxes for tracking.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        max_num = min(cls_scores.size(0), self.max_num)

        cls_scores = cls_scores.sigmoid()
        _, indexs = cls_scores.max(dim=-1)
        labels = indexs % self.num_classes

        _, bbox_index = track_scores.topk(max_num)

        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]

        scores = track_scores

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = track_scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.with_nms and img_metas is not None:
            from mmdet3d.core import xywhr2xyxyr
            from mmcv.ops import nms_bev
            boxes_for_nms = xywhr2xyxyr(
                img_metas[0]['box_type_3d'](final_box_preds[:, :], 9).bev)
            nms_mask = boxes_for_nms.new_zeros(boxes_for_nms.shape[0]) > 0
            try:
                selected = nms_bev(
                    boxes_for_nms,
                    final_scores,
                    thresh=self.nms_iou_thres)
                nms_mask[selected] = True
            except:
                print('Error in NMS', boxes_for_nms, final_scores)
                nms_mask = boxes_for_nms.new_ones(boxes_for_nms.shape[0]) > 0
        else:
            nms_mask = None

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask
            if not with_mask:
                mask = torch.ones_like(mask) > 0
            if nms_mask is not None:
                mask &= nms_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]
            bbox_index = bbox_index[mask]
            
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'track_scores': track_scores,
                'obj_idxes': obj_idxes,
                'bbox_index': bbox_index,
                'mask': mask
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, with_mask=True, img_metas=None):
        """Decode bboxes.
        """
        if isinstance(preds_dicts, list):
            task_num = len(preds_dicts)

            pred_bbox_list, pred_logits_list, task_ids_list = [], [], []
            for task_id in range(task_num):
                task_pred_dict = preds_dicts[task_id][0]
                task_pred_bbox = torch.cat(
                        (task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                         task_pred_dict['dim'][-1], task_pred_dict['rot'][-1],
                         task_pred_dict['vel'][-1]),
                        dim=-1
                    )
                task_pred_logits = task_pred_dict['cls_logits'][-1]
                pred_bbox_list.append(task_pred_bbox)
                pred_logits_list.append(task_pred_logits)

                task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
                task_ids_list.append(task_ids)
            
            all_pred_logits = torch.cat(pred_logits_list, dim=-1)  # bs * nq * 10
            all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  # bs * (task nq) * 10
            all_task_ids = torch.cat(task_ids_list, dim=-1) # bs * nq * 10

            batch_size = all_pred_logits.shape[0]
            predictions_list = []
            for i in range(batch_size):
                predictions_list.append(
                    self.decode_single_task(all_pred_logits[i], all_pred_bbox[i], all_task_ids[i]))
            return predictions_list
        else:
            # Handle tracker format (single dictionary)
            all_cls_scores = preds_dicts['cls_scores']
            all_bbox_preds = preds_dicts['bbox_preds']
            track_scores = preds_dicts.get('track_scores', None)
            obj_idxes = preds_dicts.get('obj_idxes', None)
            
            if track_scores is None:
                track_scores = all_cls_scores.sigmoid().max(dim=-1).values
            if obj_idxes is None:
                obj_idxes = torch.full_like(track_scores, -1)

            predictions_list = []
            if all_cls_scores.dim() == 2:
                # no batch size
                predictions_list.append(self.decode_single_track(
                    all_cls_scores, all_bbox_preds, track_scores, obj_idxes, with_mask, img_metas))
            elif all_cls_scores.dim() == 3:
                # with batch size
                batch_size = all_cls_scores.shape[0]
                for i in range(batch_size):
                    predictions_list.append(self.decode_single_track(
                        all_cls_scores[i], all_bbox_preds[i], track_scores[i], obj_idxes[i], with_mask, img_metas[i:i+1] if img_metas else None))
            return predictions_list