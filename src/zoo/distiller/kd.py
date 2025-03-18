import torch
import torch.nn.functional as F
from ._base_distiller import BaseDistiller
from ...core import register
from .kd_losses import kl_div_loss
from ..rtdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

__all__ = ['KD', ]

@register()
class KD(BaseDistiller):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def distill(self, student_outputs, teacher_outputs):

        losses = {}

        '''
        student_logits = student_outputs['pred_logits'].flatten(0, 1)
        teacher_logits = teacher_outputs['pred_logits'].flatten(0, 1)
        # [N, num_queries, num_classes] -> [N*num_queries, num_classes]
        '''

        # flatten
        student_logits = student_outputs['pred_logits'].flatten(1)
        teacher_logits = teacher_outputs['pred_logits'].flatten(1)
        # [N, num_queries, num_classes] -> [N, num_queries*num_classes]

        student_boxes = student_outputs['pred_boxes'].flatten(0, 1)
        teacher_boxes = teacher_outputs['pred_boxes'].flatten(0, 1)
        # [N, num_queries, 4] -> [N * num_queries, 4]

        losses['distill_cls_loss'] = kl_div_loss(student_logits, teacher_logits, self.temperature)

        num_boxes = student_boxes.shape[0]

        loss_bbox = F.l1_loss(student_boxes, teacher_boxes, reduction='none')
        losses['distill_bbox_loss'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(student_boxes), box_cxcywh_to_xyxy(teacher_boxes)))
        losses['distill_giou_loss'] = loss_giou.sum() / num_boxes

        return losses