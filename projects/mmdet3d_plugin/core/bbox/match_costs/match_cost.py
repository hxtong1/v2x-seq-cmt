import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST

try:
    from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_nearest_3d
except ImportError:
    bbox_overlaps_nearest_3d = None

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class BBoxBEVL1Cost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, pc_range):
        pc_start = bboxes.new(pc_range[0:2])
        pc_range = bboxes.new(pc_range[3:5]) - bboxes.new(pc_range[0:2])
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@MATCH_COST.register_module()
class IoU3DCost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        iou_cost = - iou
        return iou_cost * self.weight


@MATCH_COST.register_module()
class BBoxIoUBEVCost(object):
    """BEV IoU cost for Hungarian matching. Reduces AOE by favoring orientation-aligned matches.

    Args:
        weight (float): Cost weight. cost = -iou * weight (higher IoU -> lower cost).
    """

    def __init__(self, weight=0.5):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes, pc_range):
        """Compute BEV IoU cost matrix.

        Args:
            bbox_pred (Tensor): [N, 10] (cx,cy,cz,log_w,log_l,log_h,sin,cos,vx,vy).
            gt_bboxes (Tensor): [M, 9] raw (cx,cy,cz,w,l,h,rot,vx,vy).
            pc_range (list): [xmin,ymin,zmin,xmax,ymax,zmax], used if pred center is normalized.

        Returns:
            Tensor: [N, M] cost matrix (higher IoU -> lower cost).
        """
        if bbox_overlaps_nearest_3d is None:
            return bbox_pred.new_zeros(bbox_pred.shape[0], gt_bboxes.shape[0])

        # pred: build (cx,cy,log_w,log_l,cz,log_h,sin,cos) for denormalize
        pred_aligned = bbox_pred[:, [0, 1, 3, 4, 2, 5, 6, 7]]
        if bbox_pred.size(-1) > 8:
            pred_aligned = torch.cat([pred_aligned, bbox_pred[:, 8:10]], dim=-1)
        pred_phys = denormalize_bbox(pred_aligned, pc_range)  # [N, 9]
        pred_boxes_7 = pred_phys[:, :7].contiguous()  # (cx,cy,cz,w,l,h,yaw)

        # gt: [M, 9] -> [M, 7]
        gt_boxes_7 = gt_bboxes[:, :7].contiguous()

        # mmdet3d expects [x,y,z,h,w,l,ry] for lidar
        # SECOND: (x,y,z,w,l,h,yaw) -> mmdet3d: (x,y,z,h,w,l,ry)
        pred_mm = pred_boxes_7[:, [0, 1, 2, 5, 3, 4, 6]]  # (x,y,z,h,w,l,ry)
        gt_mm = gt_boxes_7[:, [0, 1, 2, 5, 3, 4, 6]]

        with torch.no_grad():
            iou = bbox_overlaps_nearest_3d(pred_mm, gt_mm, mode='iou', is_aligned=False)
        # iou [N, M]; cost = -iou * weight
        return (-iou * self.weight).clamp(min=-1e6)