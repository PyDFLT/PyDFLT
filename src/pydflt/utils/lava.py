from torch import nn
import torch

class LAVA(nn.Module):
    def __init__(self, optmodel, threshold: float):
        super().__init__()
        self.optmodel = optmodel
        self.threshold = threshold

    def forward(self, cp, adj_verts, w_rel, mm):
        diffs = (adj_verts - w_rel.unsqueeze(1)) * mm * cp.unsqueeze(1)
        obj_diffs = torch.sum(diffs, dim=-1)
        obj_diffs[obj_diffs < self.threshold] = 0

        is_padding = torch.all(adj_verts == 0, dim=-1)
        obj_diffs[is_padding] = 0

        return obj_diffs.mean()