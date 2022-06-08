import torch
import torch.nn.functional as F
from torch import nn

from fix_utils.train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def consistency_loss(logits_w, logits_s, logits_pesudo, iters=0, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True,
                     use_mix_labels=False):
    assert name in ['ce', 'L2', 'bce']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'bce':
        softmax = nn.Softmax(dim=1)
        bce = nn.BCEWithLogitsLoss()

        b, n = logits_w.size()
        prob_w = softmax(logits_w)
        prob_s = softmax(logits_s)

        # Similarity in output space
        similarity = torch.bmm(prob_w.view(b, 1, n), prob_s.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        return bce(similarity, ones)

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()

        if use_mix_labels & iters < 20000:
            one_weight = (0.5 + torch.div(max_probs - p_cutoff, 1 - p_cutoff) * 0.5).unsqueeze(dim=1)
            max_one_hot = torch.zeros_like(logits_w).float().scatter_(1, max_idx.unsqueeze(dim=1), one_weight)
            logits_pesudo_onehot = torch.zeros_like(logits_w).float().scatter_(1, logits_pesudo.unsqueeze(dim=1),
                                                                               1 - one_weight)
            mix_label = one_weight * max_one_hot + (1 - one_weight) * logits_pesudo_onehot
            masked_loss = ce_loss(logits_s, mix_label, False, reduction='none') * mask
        elif use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')
