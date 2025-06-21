import torch
import torch.nn as nn
import numpy as np
from ..rtdetr.box_ops import box_cxcywh_to_xyxy
from torchvision.ops.boxes import batched_nms
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized

import torch.nn.functional as F

from ...core import register

__all__ = ['DetaResetter', 'PredResetter', 'BlankResetter', 'Remap2CocoResetter', 'cdnPseudoResetter']

@register()
class DetaResetter(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    def forward(self, teacher_outputs, student_num_queries):
        out_logits, out_bbox = teacher_outputs['pred_logits'], teacher_outputs['pred_boxes']

        remap_to_80 = torch.tensor([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],
            dtype=torch.int64)

        out_logits = out_logits[:, :, remap_to_80]
        max_logits, _ = out_logits.max(dim=-1)
        max_inds = max_logits.topk(student_num_queries, dim=-1).indices
        bs = out_logits.shape[0]
        batch_idx = torch.arange(bs, device=out_logits.device)[:, None]
        teacher_outputs['pred_logits'] = out_logits[batch_idx, max_inds, :]
        teacher_outputs['pred_boxes'] = out_bbox[batch_idx, max_inds, :]

        if "aux_outputs" in teacher_outputs:
            aux_outputs = teacher_outputs["aux_outputs"]
            for i, aux_output in enumerate(aux_outputs):
                aux_out_logits, aux_out_bbox = aux_output['pred_logits'], aux_output['pred_boxes']
                aux_out_logits = aux_out_logits[:, :, remap_to_80]
                aux_out_logits = aux_out_logits[batch_idx, max_inds, :]
                aux_out_bbox = aux_out_bbox[batch_idx, max_inds, :]
                aux_outputs[i] = {'pred_logits': aux_out_logits, 'pred_boxes': aux_out_bbox}

        return teacher_outputs

@register()
class PredResetter(nn.Module):
    def __init__(self, num_student_decoder_layers=3, num_teacher_decoder_layers=6) -> None:
        super().__init__()
        self.num_student_decoder_layers = num_student_decoder_layers
        self.num_teacher_decoder_layers = num_teacher_decoder_layers

    def forward(self, teacher_outputs):
        assert self.num_teacher_decoder_layers%self.num_student_decoder_layers == 0
        if self.num_teacher_decoder_layers == self.num_student_decoder_layers:
            return teacher_outputs
        else:
            decoder_outputs = teacher_outputs['aux_outputs']
            decoder_outputs.append({'pred_logits': teacher_outputs['pred_logits'], 'pred_boxes': teacher_outputs['pred_boxes']})
            quotient_num = self.num_teacher_decoder_layers//self.num_student_decoder_layers

            new_outputs = []

            for i in range(self.num_student_decoder_layers):
                start_idx = i * quotient_num
                end_idx = (i + 1) * quotient_num

                group_logits = [decoder_outputs[idx]['pred_logits'] for idx in range(start_idx, end_idx)]

                group_boxes = [decoder_outputs[idx]['pred_boxes'] for idx in range(start_idx, end_idx)]

                avg_logits = torch.stack(group_logits, dim=0).mean(dim=0)
                avg_boxes = torch.stack(group_boxes, dim=0).mean(dim=0)

                new_outputs.append({
                    'pred_logits': avg_logits,
                    'pred_boxes': avg_boxes
                })

            teacher_outputs = {}
            teacher_outputs['aux_outputs'] = new_outputs[:-1]
            teacher_outputs['pred_logits'] = new_outputs[-1]['pred_logits']
            teacher_outputs['pred_boxes'] = new_outputs[-1]['pred_boxes']

            return teacher_outputs

@register()
class BlankResetter(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    def forward(self, teacher_outputs):
        return teacher_outputs

@register()
class Remap2CocoResetter(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, teacher_outputs):
        teacher_outputs['pred_logits'] = self.remap(teacher_outputs['pred_logits'])
        enc_outputs = teacher_outputs['enc_aux_outputs']
        enc_outputs[0]['pred_logits'] = self.remap(enc_outputs[0]['pred_logits'])
        aux_outputs = teacher_outputs['aux_outputs']
        for i in range(len(teacher_outputs['aux_outputs'])):
            aux_outputs[i]['pred_logits'] = self.remap(aux_outputs[i]['pred_logits'])

        return teacher_outputs

    def remap(self, logits):
        remap_to_80 = torch.tensor([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],
            dtype=torch.int64)
        logits = logits[:, :, remap_to_80]
        return logits

@register()
class cdnPseudoResetter(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.threshold = 0.5  # threshold for pseudo label generation

    def forward(self, teacher_outputs):
        teacher_logits= teacher_outputs['pred_logits']
        score = F.sigmoid(teacher_logits)
        score_max, score_argmax = score.max(dim=-1)

        valid_mask = score_max > self.threshold

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
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=teacher_logits.device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        meta = {'num_boxes': num_boxes,
                'pseudo_targets': pseudo_targets, }
        return meta

    def set_epoch(self, epoch):
        """Descent the threshold for pseudo label generation."""
        orig_thres = self.threshold
        if epoch <= 36:
            self.threshold = 0.5
        elif epoch <= 72:
            self.threshold = 0.35

        if self.threshold != orig_thres:
            print(f"Threshold for pseudo label generation changed from {orig_thres} to {self.threshold} at epoch {epoch}.")