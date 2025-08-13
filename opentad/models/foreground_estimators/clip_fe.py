import os
import copy
import numpy as np

from ..builder import FOREGROUND_ESTIMATORS
from ..bricks import ConvModule

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSingleProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="relu"),
        drop_out=None,
    ):
        super().__init__()
        assert num_convs > 0
        self.drop_out = nn.Dropout1d(p=drop_out) if drop_out is not None else None

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def forward(self, x, mask):
        # x shape [B,C,T], mask [B,T]

        if self.drop_out is not None:
            x = self.drop_out(x)

        for conv in self.convs:
            x, mask = conv(x, mask)
        return x, mask



@FOREGROUND_ESTIMATORS.register_module()
class ClipForegroundEstimator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, num_layers=2, drop_out=0.3, r_act=8, num_classes=20):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.drop_out = drop_out
        self.r_act = r_act
        self.num_classes = num_classes

        conv_cfg=dict(kernel_size=1, padding=0)
        norm_cfg=dict(type="GN", num_groups=32)
        act_cfg=None

        self.feature_embedding = ConvSingleProj(
            in_channels=self.feature_dim,
            out_channels=self.out_dim,
            num_convs=self.num_layers,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.cls_layer = nn.Linear(self.out_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

        self.bce_criterion = nn.BCELoss(reduction='none')
    
    def forward(self, input_features, masks, text_proto, img_feats, img_masks, gt_labels=None):
        vid_labels = self._prepare_targets(gt_labels)
        text_proto = text_proto[0]

        B, F, T = input_features.shape  # [B,F,T]
        # input_features = input_features.permute(0, 2, 1)  # [B,F,T]
        feature_embedding, _ = self.feature_embedding(input_features, masks)

        feature_embedding = feature_embedding.transpose(1, 2)  # [B,T,F]
        cls_preds = self.cls_layer(feature_embedding)  # [B,T,C]
        cls_logits = self.sigmoid(cls_preds)  # [B,T,C]

        # >> clip logits
        img_feats = [img_feat.unsqueeze(0) for img_feat in img_feats]
        img_masks = [img_mask.unsqueeze(0) for img_mask in img_masks]
        img_feats = torch.cat(img_feats)
        img_masks = torch.cat(img_masks)
        text_proto = text_proto.transpose(0, 1).to(feature_embedding.dtype)

        text_logits = torch.matmul(img_feats, text_proto)

        text_vid_scores = []
        for i in range(B):
            text_logit = text_logits[i]
            img_mask = img_masks[i]
            valid_len = img_mask.sum().item()
            k_act = max(1, valid_len // self.r_act)
            text_value, _ = text_logit.sort(descending=True, dim=0)
            text_topk_scores = text_value[:k_act, :]
            text_vid_scores.append(torch.mean(text_topk_scores, dim=0).unsqueeze(0))
        text_vid_scores = torch.cat(text_vid_scores)

        cls_vid_scores = []
        for i in range(B):
            cls_logit = cls_logits[i]
            mask = masks[i]
            valid_len = mask.sum().item()
            k_act = max(1, valid_len // self.r_act)
            cls_value, _ = cls_logit.sort(descending=True, dim=0)
            cls_topk_scores = cls_value[:k_act, :]
            cls_vid_scores.append(torch.mean(cls_topk_scores, dim=0).unsqueeze(0))
        cls_vid_scores = torch.cat(cls_vid_scores)

        if vid_labels is None:
            vid_scores = text_vid_scores
        else:
            vid_scores = cls_vid_scores
        
        if vid_labels is not None:
            fe_loss_dict = {}
            video_loss_dict = self.cls_video_criterion(
                vid_score=vid_scores,
                vid_label=vid_labels,
            )
            fe_loss_dict['loss_vid_text'] = video_loss_dict['loss_video_vid']

            return fe_loss_dict
        else:
            return vid_scores

    def _prepare_targets(self, gt_labels):
        """
        Prepare targets for training.
        """
        if gt_labels is None:
            return None
        else:
            now_device = gt_labels[0].device
            gt_nums = len(gt_labels)
            vid_labels = torch.zeros((gt_nums, self.num_classes)).to(now_device)
            for ggl in range(len(gt_labels)):
                gt_global_tmp = gt_labels[ggl].cpu().detach().numpy().tolist()
                for ggt in gt_global_tmp:
                    vid_labels[ggl][ggt] = 1.
            return vid_labels

    def cls_video_criterion(
            self,
            vid_score,
            vid_label,
        ):
        loss_vid = self.bce_criterion(vid_score, vid_label)
        loss_vid = loss_vid.mean()
        loss_dict = {}
        loss_dict["loss_video_vid"] = loss_vid * 10.0

        return loss_dict

