import torch
import torch.nn as nn

class BaseDistiller(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.need_matcher = False

    #TODO: If the base distiller is not useful, remove it in later stage.
    def forward(self,student_outputs, teacher_outputs, meta=None):
        if self.need_matcher:
            losses = self.match_distill(student_outputs, teacher_outputs, meta)
        else:
            losses = self.distill(student_outputs, teacher_outputs, meta)
        return losses

    def distill(self, student_outputs, teacher_outputs, meta=None):
        raise NotImplementedError('')