import torch
import torch.nn.functional as F



def kl_div_loss(logits_student, logits_teacher, temperature, nqc=False):
    if nqc:
        n, q, c = logits_student.shape
        logits_student = logits_student.flatten(0, 1)
        logits_teacher = logits_teacher.flatten(0, 1)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none")
    if nqc:
        loss_kd = loss_kd.view(n, q, c)
        loss_kd = loss_kd.mean(1).sum()
    else:
        loss_kd = loss_kd.mean(0).sum()
    loss_kd *= temperature**2
    return loss_kd