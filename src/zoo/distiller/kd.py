import torch
import torch.nn.functional as F
import torchvision
from ._base_distiller import BaseDistiller
from ...core import register
from .kd_losses import kl_div_loss
from ..rtdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized

__all__ = (['KD', ])

@register()
class KD(BaseDistiller):
    def __init__(self, temperature=2.0, logits_loss_type=None,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.logits_loss_type = logits_loss_type
        self.alpha = alpha
        self.gamma = gamma

    def distill(self, student_outputs, teacher_outputs, meta=None):

        losses = {}


        student_logits = student_outputs['pred_logits']
        teacher_logits = teacher_outputs['pred_logits']

        if meta is None:

            threshold = 0.5

            #find the indices of the positive samples
            score = F.sigmoid(teacher_logits)
            score_max, score_argmax = score.max(dim=-1)

            valid_mask = score_max > threshold

            indices = valid_mask.nonzero(as_tuple=False)

            num_boxes = valid_mask.sum()
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=student_logits.device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            meta = {'indices': indices, 'num_boxes': num_boxes, 'valid_mask': valid_mask, 'score_argmax': score_argmax}

        else:
            indices = meta['indices']
            num_boxes = meta['num_boxes']
            valid_mask = meta['valid_mask']
            score_argmax = meta['score_argmax']

        batch_idx = indices[:, 0]
        query_idx = indices[:, 1]

        distill_student_logits = student_logits[batch_idx, query_idx]  # [M, C]
        distill_teacher_logits = teacher_logits[batch_idx, query_idx]  # [M, C]

        distill_student_boxes = student_outputs['pred_boxes'][batch_idx, query_idx]  # [M, 4]
        distill_teacher_boxes = teacher_outputs['pred_boxes'][batch_idx, query_idx]  # [M, 4]

        if self.logits_loss_type == 'vanilla':
            cls_loss = kl_div_loss(distill_student_logits, distill_teacher_logits,
                                                     self.temperature)

            losses['distill_cls_loss'] = cls_loss.sum() / num_boxes

        elif self.logits_loss_type == 'sigbce':
            teacher_probs = torch.sigmoid(distill_teacher_logits)  # 变成 [0,1] 之间

            ce_loss = F.binary_cross_entropy_with_logits(distill_student_logits,
                                                         teacher_probs, reduction='none')
            losses['distill_cls_loss'] = ce_loss.sum() / num_boxes

        elif self.logits_loss_type == 'l1':
            loss_cls = F.l1_loss(distill_student_logits, distill_teacher_logits, reduction='none')
            losses['distill_cls_loss'] = loss_cls.sum() / num_boxes

        elif self.logits_loss_type == 'focal':
            one_hot_teacher_logits = F.one_hot(score_argmax, num_classes=student_logits.shape[-1] + 1)[..., :-1].to(
                distill_student_logits.dtype)

            one_hot_distill_teacher_logits = one_hot_teacher_logits[batch_idx, query_idx]

            loss = torchvision.ops.sigmoid_focal_loss(distill_student_logits, one_hot_distill_teacher_logits,
                                                      self.alpha, self.gamma, reduction='none')
            loss = loss.sum() / num_boxes

            losses['distill_cls_loss'] = loss

        elif self.logits_loss_type == 'vfl':

            src_boxes = distill_student_boxes
            target_boxes = distill_teacher_boxes
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()

            src_logits = student_outputs['pred_logits']

            target = torch.zeros_like(student_outputs['pred_logits'])  # [N,Q,C]
            if valid_mask.any():
                target[valid_mask] = F.one_hot(
                    score_argmax[valid_mask], num_classes=src_logits.shape[-1] + 1)[..., :-1].to(target.dtype)
            # [N, num_queries, num_classes]

            target_score_o = torch.zeros_like(valid_mask, dtype=src_logits.dtype)
            target_score_o[valid_mask] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target
            # [N, num_queries, num_classes]

            pred_score = F.sigmoid(src_logits).detach()
            weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

            losses['distill_cls_loss'] = loss

        # box loss
        loss_bbox = F.l1_loss(distill_student_boxes, distill_teacher_boxes, reduction='none')
        losses['distill_bbox_loss'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(distill_student_boxes), box_cxcywh_to_xyxy(distill_teacher_boxes)))
        losses['distill_giou_loss'] = loss_giou.sum() / num_boxes

        return losses, meta
