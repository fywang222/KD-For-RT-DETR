import torch
import torch.nn.functional as F
from ._base_distiller import BaseDistiller
from ...core import register
from ..rtdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized

__all__ = (['Hungarian_KD', ])

@register()
class Hungarian_KD(BaseDistiller):
    def __init__(self, share_matched_idx=False):
        super().__init__()
        self.share_matched_idx = share_matched_idx
        self.threshold = 0.5  # threshold for pseudo label generation
    def distill(self, student_outputs, teacher_meta, ):
        assert teacher_meta is not None, "Do you really want to use hungarian matching?"

        losses = {}

        matcher = teacher_meta['matcher']
        num_boxes = teacher_meta['num_boxes']

        pseudo_targets = teacher_meta['pseudo_targets']

        if self.share_matched_idx and 'indices' in teacher_meta:
            indices = teacher_meta['indices']
        else:
            indices = matcher(student_outputs, pseudo_targets)['indices']

        losses['distill_cls_loss'] = self._vfl_loss(student_outputs, pseudo_targets, indices, num_boxes)

        losses.update(self.loss_boxes(student_outputs, pseudo_targets, indices, num_boxes,))

        return losses, teacher_meta

    def _vfl_loss(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        num_classes = src_logits.shape[-1]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=num_classes + 1)[..., :-1]
        # [N, num_queries, num_classes]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        # [N, num_queries, num_classes]

        pred_score = F.sigmoid(src_logits).detach()
        weight = 0.75 * pred_score.pow(2.0) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return loss

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
