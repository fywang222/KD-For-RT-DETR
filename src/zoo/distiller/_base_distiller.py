import torch
import torch.nn as nn

class BaseDistiller(nn.Module):
    def __init__(self, ):
        super().__init__()

    #TODO: If the base distiller is not useful, remove it in later stage.
    def forward(self,student_outputs, teacher_meta):

        losses, meta = self.distill(student_outputs, teacher_meta)
        return losses, meta

    def distill(self, student_outputs, teacher_meta):
        raise NotImplementedError('')