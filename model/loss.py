

import torch.nn.functional as F

def xent_loss(output, target):
    return F.CrossEntropyLoss(output, target)