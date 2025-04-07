import torch
import torch.nn.functional as F
import torchvision
from ._base_distiller import BaseDistiller
from ...core import register
from .kd_losses import kl_div_loss
from ..rtdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized

__all__ = ['KD', ]

@register()
class KD(BaseDistiller):
    def __init__(self, temperature=2.0, logits_loss_type=None, one_hot=False,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.logits_loss_type = logits_loss_type
        self.one_hot = one_hot
        self.alpha = alpha
        self.gamma = gamma

    def distill(self, student_outputs, teacher_outputs):

        losses = {}

        student_logits = student_outputs['pred_logits']
        teacher_logits = teacher_outputs['pred_logits']

        if self.one_hot:
            score = F.sigmoid(teacher_logits)
            score_max, score_argmax = score.max(-1)
            mask = score_max > 0.5
            score_argmax = torch.where(mask, score_argmax, torch.tensor(score.shape[-1], device=score.device))

            one_hot_target = F.one_hot(score_argmax, num_classes=score.shape[-1] + 1)
            one_hot_target = one_hot_target[:, :, :-1]
            teacher_logits = one_hot_target.to(teacher_logits.dtype)

            num_boxes = mask.sum()
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=student_logits.device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if self.logits_loss_type == 'vanilla':
            losses['distill_cls_loss'] = kl_div_loss(student_logits, teacher_logits, self.temperature, nqc=True)

        elif self.logits_loss_type =='flatten':
            student_logits = student_logits.flatten(1)
            teacher_logits = teacher_logits.flatten(1)
            # [N, num_queries, num_classes] -> [N, num_queries*num_classes]

            losses['distill_cls_loss'] = kl_div_loss(student_logits, teacher_logits, self.temperature)

        elif self.logits_loss_type == 'sigbce':
            ce_loss = F.binary_cross_entropy_with_logits(F.sigmoid(student_logits), F.sigmoid(teacher_logits),
                                                            reduction='mean')
            losses['distill_cls_loss'] = ce_loss

        elif self.logits_loss_type == 'l1':
            loss_cls = F.l1_loss(student_logits, teacher_logits, reduction='mean')
            losses['distill_cls_loss'] = loss_cls

        elif self.logits_loss_type == 'focal':
            assert self.one_hot == True;

            # TODO: add the parameter alpha and gamma to the config file
            loss = torchvision.ops.sigmoid_focal_loss(student_logits, teacher_logits,
                                                      self.alpha, self.gamma, reduction='none')
            loss = loss.mean(1).sum() * student_logits.shape[1] / num_boxes

            losses['distill_cls_loss'] = loss

        student_boxes = student_outputs['pred_boxes'].flatten(0, 1)
        teacher_boxes = teacher_outputs['pred_boxes'].flatten(0, 1)
        # [N, num_queries, 4] -> [N * num_queries, 4]

        num_boxes = student_boxes.shape[0]

        loss_bbox = F.l1_loss(student_boxes, teacher_boxes, reduction='none')
        losses['distill_bbox_loss'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(student_boxes), box_cxcywh_to_xyxy(teacher_boxes)))
        losses['distill_giou_loss'] = loss_giou.sum() / num_boxes

        return losses