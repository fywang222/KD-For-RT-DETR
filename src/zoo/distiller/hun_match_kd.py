import torch
import torch.nn.functional as F
from ._base_distiller import BaseDistiller
from ...core import register
from ..rtdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized

__all__ = (['Hungarian_KD', ])

@register()
class Hungarian_KD(BaseDistiller):
    def __init__(self, ):
        super().__init__()
    def distill(self, student_outputs, teacher_outputs, meta=None, shared_matched_idx=False):

        losses = {}

        student_logits = student_outputs['pred_logits']
        teacher_logits = teacher_outputs['pred_logits']

        matcher = meta['matcher']
        num_boxes = meta['num_boxes']

        if num_boxes is None:

            threshold = 0.5

            # find the indices of the positive samples
            score = F.sigmoid(teacher_logits)
            score_max, score_argmax = score.max(dim=-1)

            valid_mask = score_max > threshold

            pseudo_targets = []
            for b in range(teacher_logits.shape[0]):
                q_idxs = valid_mask[b].nonzero(as_tuple=False).squeeze(1)

                labels = score_argmax[b, q_idxs]

                boxes = teacher_outputs['pred_boxes'][b, q_idxs]

                pseudo_targets.append({
                    "labels": labels,  # Tensor[M_b]
                    "boxes": boxes  # Tensor[M_b, 4]
                })

            num_boxes = valid_mask.sum()
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=student_logits.device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            indices = matcher(student_outputs, pseudo_targets)['indices']
            meta = {'matcher': matcher,'num_boxes': num_boxes,
                    'pseudo_targets': pseudo_targets, 'indices': indices,}

        else:
            pseudo_targets = meta['pseudo_targets']
            if shared_matched_idx:
                indices = meta['indices']
            else:
                indices = matcher(student_outputs, pseudo_targets)['indices']

        losses['distill_cls_loss'] = self._vfl_loss(student_outputs, pseudo_targets, indices, num_boxes)

        losses.update(self.loss_boxes(student_outputs, pseudo_targets, indices, num_boxes,))

        return losses, meta

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