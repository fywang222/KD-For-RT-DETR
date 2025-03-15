import torch
import torch.nn as nn

class BaseDistiller(nn.Module):
    def __init__(self, ):
        super().__init__()

    #TODO: If the base distiller is not useful, remove it in later stage.
    def forward(self,student_outputs, teacher_outputs):
        losses = self.distill(student_outputs, teacher_outputs)
        return losses

    def distill(self, student_outputs, teacher_outputs, **kwargs):
        raise NotImplementedError('')