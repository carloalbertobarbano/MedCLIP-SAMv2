"""
Author: Carlo Alberto Barbano (carlo.barbano@unito.it)
Date: 12/05/24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, target_features):
        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            # .permute(1, 0, 2) works with ViT (feats shape [num_patches, batch_size, feat_dim])
            input_feat = F.normalize(input_feat, p=2, dim=-1)
            target_feat = F.normalize(target_feat, p=2, dim=-1)

            gram_input = input_feat.permute(1, 0, 2) @ input_feat.permute(1, 0, 2).transpose(1, 2)
            gram_target = target_feat.permute(1, 0, 2) @ target_feat.permute(1, 0, 2).transpose(1, 2)
            loss += F.mse_loss(gram_input, gram_target)
        return loss / len(input_features)
