import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale
from ..utils.post_processing import batched_nms, convert_to_seconds


class FeatureProj(nn.Module):
    def __init__(self, in_channel=512, hidden_channel=256, out_channel=512):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.proj = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, out_channel),
        )

    def forward(self, x):
        proj_x = self.proj(x)
        # proj_x = F.normalize(proj_x, dim=-1)
        out_x = x + proj_x
        out_x_norm = F.normalize(out_x, dim=-1)
        return out_x_norm


@HEADS.register_module()
class AnchorFreeHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        filter_similar_gt=True,
        neg_num=3,
        score_thres=0.1,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt

        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))  # save in the state_dict

        # point generator
        self.prior_generator = build_prior_generator(prior_generator)

        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

        self.feat_proj = FeatureProj()
        self.contrast_loss = nn.CrossEntropyLoss()
        self.neg_num = neg_num
        self.score_thres = score_thres

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_heads()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_heads(self):
        """Initialize predictor layers of the head."""
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        contrast_loss = None
        clip_image_feat = kwargs.get("clip_image_feat", None)
        if clip_image_feat is not None:
            clip_text_feat = kwargs.get("clip_text_feat", None)
            gt_real_labels = kwargs.get("gt_real_labels", None)
            class_map_idx = kwargs.get("class_map_idx", None)
            # get refined proposals and scores
            proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]

            visual_feats = []
            text_feats = []

            real_segments = []
            for pp_idx in range(len(proposals)):
                segments = proposals[pp_idx]
                seg_meta = metas[pp_idx]
                real_segs = convert_to_seconds(segments, seg_meta)
                real_segments.append(real_segs)

                item_image_feat = clip_image_feat[pp_idx]
                item_text_feat = clip_text_feat[pp_idx]
                item_class_map_idx = class_map_idx[pp_idx]

                gt_segs = gt_segments[pp_idx]
                real_gt_segs = convert_to_seconds(gt_segs, seg_meta)
                gt_lbls = gt_real_labels[pp_idx]

                feat_len_min = 0
                feat_len_max = item_image_feat.shape[0] - 1

                pred_clip_scores = []
                for (start, end), gt_lbl in zip(real_gt_segs, gt_lbls):
                    if gt_lbl.item() == -1:
                        continue
                    start_idx = int(start * 10)
                    end_idx = int(end * 10)
                    start_idx = max(feat_len_min, start_idx)
                    end_idx = min(end_idx, feat_len_max)
                    feature_segment = item_image_feat[start_idx-1:end_idx+1]
                    feature_segment = feature_segment.mean(dim=0, keepdim=True)

                    if torch.isnan(feature_segment.max()):
                        continue

                    visual_feats.append(feature_segment)
                    text_feats.append(item_text_feat[gt_lbl].unsqueeze(0))

                    ignore_id = gt_lbl.item()
                    valid_indices = []
                    for icmi in item_class_map_idx:
                        if icmi.item() != ignore_id:
                            valid_indices.append(icmi.item())
                    neg_indices = random.sample(valid_indices, self.neg_num)
                    for neg_lbl in neg_indices:
                        text_feats.append(item_text_feat[neg_lbl].unsqueeze(0))

                    # clip_scores = feature_segment @ item_text_feat[gt_lbl].unsqueeze(0).T
                    # pred_clip_scores.append(clip_scores)
            
            if len(visual_feats) > 0:
                visual_feats = torch.cat(visual_feats)
                text_feats = torch.cat(text_feats)
                N = visual_feats.shape[0]
                text_feats_reshaped = text_feats.view(N, self.neg_num+1, -1)

                visual_feats = self.feat_proj(visual_feats)
                visual_feats_expanded = visual_feats.unsqueeze(1)

                similarity_matrix = (visual_feats_expanded @ text_feats_reshaped.transpose(1, 2)).squeeze(1)
                contrast_labels = torch.zeros(similarity_matrix.shape[0], dtype=torch.long).to(similarity_matrix.device)
                contrast_loss = self.contrast_loss(similarity_matrix, contrast_labels)

        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        if contrast_loss is not None:
            losses['contrast_loss'] = contrast_loss
        return losses

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]

        vll_classes = kwargs.get("vll_classes", None)
        if vll_classes is not None:
            vll_scores = kwargs.get("vll_scores", None)
            clip_text_feat = kwargs.get("clip_text_feat", None)
            clip_image_feat = kwargs.get("clip_image_feat", None)

            segment_labels = []
            for pp_idx in range(len(proposals)):
                segments = proposals[pp_idx]
                seg_meta = metas[pp_idx]
                real_segs = convert_to_seconds(segments, seg_meta)
                
                item_image_feat = clip_image_feat[pp_idx]
                item_text_feat = clip_text_feat[pp_idx]
                item_vll_classes = vll_classes[pp_idx]
                item_vll_scores = vll_scores[pp_idx]

                visual_feats = []
                text_feats = []

                feat_len_min = 0
                feat_len_max = item_image_feat.shape[0] - 1

                for start, end in real_segs:
                    start_idx = int(start * 10)
                    end_idx = int(end * 10)
                    start_idx = max(feat_len_min, start_idx)
                    end_idx = min(end_idx, feat_len_max)
                    feature_segment = item_image_feat[start_idx:end_idx]
                    feature_segment = feature_segment.mean(dim=0, keepdim=True)

                    visual_feats.append(feature_segment)
                    for vll_lbl in item_vll_classes:
                        text_feats.append(item_text_feat[vll_lbl].unsqueeze(0))

                visual_feats = torch.cat(visual_feats)
                text_feats = torch.cat(text_feats)
                N = visual_feats.shape[0]
                text_feats_reshaped = text_feats.view(N, 2, -1)

                visual_feats = self.feat_proj(visual_feats)
                visual_feats_expanded = visual_feats.unsqueeze(1)
                similarity_matrix = (visual_feats_expanded @ text_feats_reshaped.transpose(1, 2)).squeeze(1)

                segment_label = []
                for i in range(similarity_matrix.shape[0]):
                    seg_pred = similarity_matrix[i]
                    seg_pred_score_0 = seg_pred[0].item()
                    seg_pred_score_1 = seg_pred[1].item()
                    if abs(seg_pred_score_0 - seg_pred_score_1) >= self.score_thres:
                        if seg_pred_score_0 > seg_pred_score_1:
                            segment_label.append([item_vll_classes[0], item_vll_scores[0]])
                        else:
                            segment_label.append([item_vll_classes[1], item_vll_scores[1]])
                    else:
                        segment_label.append([item_vll_classes[0], item_vll_classes[1], item_vll_scores[0], item_vll_scores[1]])

                segment_labels.append(segment_label)

        return proposals, scores, segment_labels

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=0)  # [T,4]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]

        start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
        end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)  # [B,T,2]
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        # apply regression to get refined proposals
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        # proposal scores
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,num_classes]

        # mask out invalid, and return a list with batch size
        masks = torch.cat(mask_list, dim=1)  # [B,T]
        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])  # [T,2]
            new_scores.append(score[mask])  # [T,num_classes]
        return new_proposals, new_scores

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            # giou loss defined on positive samples
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        gt_cls, gt_reg = [], []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            num_gts = gt_segment.shape[0]

            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            # compute the lengths of all segments -> F T x N
            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # concat_points[:, 3] stores the stride
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                # F T x N x 2
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                # F T x N
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                # inside an gt action
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            # limit the regression range for each location
            max_regress_distance = reg_targets.max(-1)[0]
            # F T x N
            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
            )

            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)

            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)
            # OK to use min_len_inds
            reg_targets = reg_targets[range(num_pts), min_len_inds]
            # normalization based on stride
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg
