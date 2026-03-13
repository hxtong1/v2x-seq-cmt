# ------------------------------------------------------------------------
# BEV IoU Loss for CMT query-transformer. Corresponds to BBoxIoUBEVCost in
# Hungarian matching. Uses differentiable GIoU on BEV AABB (axis-aligned
# bounding box of rotated rect).
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


def _bev_boxes_to_xyxy(bev_boxes, pc_range):
    """Convert BEV boxes (cx,cy,w,l,yaw) to axis-aligned xyxy for differentiable GIoU.

    Args:
        bev_boxes (Tensor): [N, 5] (cx, cy, w, l, yaw) in physical coords.
        pc_range: unused, for API consistency.

    Returns:
        Tensor: [N, 4] (x1, y1, x2, y2) axis-aligned bounding box.
    """
    cx = bev_boxes[..., 0:1]
    cy = bev_boxes[..., 1:2]
    w = bev_boxes[..., 2:3].clamp(min=1e-4)
    l = bev_boxes[..., 3:4].clamp(min=1e-4)
    yaw = bev_boxes[..., 4:5]

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    hw = w / 2
    hl = l / 2

    # Four corners in local frame: (±hw, ±hl)
    corners_x = torch.stack([hw, -hw, -hw, hw], dim=-1)
    corners_y = torch.stack([hl, hl, -hl, -hl], dim=-1)
    # Rotate and translate
    corners_x_rot = corners_x * cos_yaw - corners_y * sin_yaw
    corners_y_rot = corners_x * sin_yaw + corners_y * cos_yaw
    corners_x_world = corners_x_rot + cx
    corners_y_world = corners_y_rot + cy

    x1 = corners_x_world.min(dim=-1, keepdim=True)[0]
    y1 = corners_y_world.min(dim=-1, keepdim=True)[0]
    x2 = corners_x_world.max(dim=-1, keepdim=True)[0]
    y2 = corners_y_world.max(dim=-1, keepdim=True)[0]
    return torch.cat([x1, y1, x2, y2], dim=-1)


def _giou_loss_2d(pred_xyxy, target_xyxy, eps=1e-7):
    """Differentiable GIoU loss for 2D boxes. pred/target: [N, 4] (x1,y1,x2,y2)."""
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_xyxy.unbind(-1)
    target_x1, target_y1, target_x2, target_y2 = target_xyxy.unbind(-1)

    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = pred_area + target_area - inter_area + eps
    iou = inter_area / union_area

    enclosing_x1 = torch.min(pred_x1, target_x1)
    enclosing_y1 = torch.min(pred_y1, target_y1)
    enclosing_x2 = torch.max(pred_x2, target_x2)
    enclosing_y2 = torch.max(pred_y2, target_y2)
    enclosing_area = (enclosing_x2 - enclosing_x1).clamp(min=0) * (
        enclosing_y2 - enclosing_y1
    ).clamp(min=0) + eps

    giou = iou - (enclosing_area - union_area) / enclosing_area
    loss = 1 - giou
    return loss


@LOSSES.register_module()
class BBoxIoUBEVLoss(nn.Module):
    """BEV IoU (GIoU) loss, corresponding to BBoxIoUBEVCost in Hungarian assigner.

    Uses differentiable GIoU on BEV axis-aligned bounding boxes of rotated rects.
    Applied only on positive (matched) samples.

    Args:
        loss_weight (float): Loss weight. Default 0.25 (aligned with iou_cost weight).
    """

    def __init__(self, loss_weight=0.25):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        pc_range=None,
    ):
        """Compute BEV IoU loss.

        Args:
            pred (Tensor): [N, 10] normalized (cx,cy,log_w,log_l,cz,log_h,sin,cos,vx,vy).
            target (Tensor): [N, 9 or 10] normalized target (same layout, center physical).
            weight (Tensor, optional): [N] positive mask.
            avg_factor (int): Number of positive samples.
            pc_range (list): [xmin,ymin,zmin,xmax,ymax,zmax] for pred center denorm.
        """
        if pred.numel() == 0 or (avg_factor is not None and avg_factor == 0):
            return pred.sum() * 0.0

        # pred layout: (cx,cy,log_w,log_l,cz,log_h,sin,cos,vx,vy)
        pred_aligned = pred[:, [0, 1, 3, 4, 2, 5, 6, 7]]
        if pred.size(-1) > 8:
            pred_aligned = torch.cat([pred_aligned, pred[:, 8:10]], dim=-1)
        pred_phys = denormalize_bbox(pred_aligned, pc_range)
        # pred center from decoder is typically [0,1]; convert to physical
        if pc_range is not None:
            pr = pred.new_tensor(pc_range)
            pred_phys = pred_phys.clone()
            pred_phys[:, 0:1] = pred_phys[:, 0:1] * (pr[3] - pr[0]) + pr[0]
            pred_phys[:, 1:2] = pred_phys[:, 1:2] * (pr[4] - pr[1]) + pr[1]
            pred_phys[:, 2:3] = pred_phys[:, 2:3] * (pr[5] - pr[2]) + pr[2]

        # target: normalized (cx,cy,log_w,log_l,cz,log_h,sin,cos[,vx,vy])
        target_8 = target[:, :8]
        target_phys = denormalize_bbox(target_8, None)

        # BEV: (cx, cy, w, l, yaw)
        pred_bev = torch.cat(
            [
                pred_phys[:, :2],
                pred_phys[:, 3:5],
                pred_phys[:, 6:7],
            ],
            dim=-1,
        )
        target_bev = torch.cat(
            [
                target_phys[:, :2],
                target_phys[:, 3:5],
                target_phys[:, 6:7],
            ],
            dim=-1,
        )

        pred_xyxy = _bev_boxes_to_xyxy(pred_bev, pc_range)
        target_xyxy = _bev_boxes_to_xyxy(target_bev, pc_range)

        loss = _giou_loss_2d(pred_xyxy, target_xyxy)

        if weight is not None:
            loss = loss * weight
        if avg_factor is not None and avg_factor > 0:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()

        return loss * self.loss_weight
