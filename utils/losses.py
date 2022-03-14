"""Custom loss functions"""

import torch
from torch.nn import functional as F

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def mean_entropy_loss(input1, input2, smooth=1e-8):
    #revised by fangcheng
    input1 = F.softmax(input1, dim=1)
    input2 = F.softmax(input2, dim=1)
    assert input1.min() >= 0 and input1.max() <= 1 and input2.min() >= 0 and input2.max() <= 1
    assert input1.size() == input2.size()
    tmp = torch.pow(input1 * torch.log2(input1+smooth) - input2 * torch.log2(input2+smooth), 2) # N x n_class x H x W
    return torch.mean(torch.sum(tmp, dim=1))


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, softmax=False):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        if target.dim() < inputs.dim():
            target = target.unsqueeze(1)
        if target.shape[1] == 1 and target.shape[1] < inputs.shape[1]:
            target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict & target shape do not match, with inputs={inputs.shape}, target={target.shape})'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class Compose:
    def __init__(self, loss_list, weights=None):
        self.loss_list = loss_list
        self.weights = weights if weights else [1.] * len(self.loss_list)

    def __call__(self, *args, **kwargs):
        l = 0.
        for weight, loss in zip(self.weights, self.loss_list):
            l += weight * loss(*args, **kwargs)
        return l
