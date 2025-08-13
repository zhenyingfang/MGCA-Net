import torch

from ..builder import DETECTORS, build_forground_estimator
from .deformable_detr import DeformableDETR
from ..utils.bbox_tools import proposal_cw_to_se
from ..utils.post_processing import batched_nms, convert_to_seconds, load_predictions, save_predictions

import numpy as np
import json

# >> Anet-1.3
# ext_path = "/root/autodl-tmp/code/OV_TAL/OV_OpenTAL/data/activitynet-1.3/classifiers/cuhk_val_simp_7.json"
ext_path = "/root/autodl-tmp/code/OV_TAL/annotations/clip_imgs/anet_clip_score_b16.json"
with open(ext_path, "r") as f:
    cuhk_data = json.load(f)
cuhk_data_score = cuhk_data["results"]
cuhk_data_action = np.array(cuhk_data["class"])

# >> THUMOS-14
thumos_class = {
            7: "BaseballPitch",
            9: "BasketballDunk",
            12: "Billiards",
            21: "CleanAndJerk",
            22: "CliffDiving",
            23: "CricketBowling",
            24: "CricketShot",
            26: "Diving",
            31: "FrisbeeCatch",
            33: "GolfSwing",
            36: "HammerThrow",
            40: "HighJump",
            45: "JavelinThrow",
            51: "LongJump",
            68: "PoleVault",
            79: "Shotput",
            85: "SoccerPenalty",
            92: "TennisSwing",
            93: "ThrowDiscus",
            97: "VolleyballSpiking",
        }
# ext_path = "/root/autodl-tmp/code/OV_TAL/OV_OpenTAL/data/thumos-14/classifiers/uNet_test.npy"
ext_path = "/root/autodl-tmp/code/OV_TAL/annotations/clip_imgs/thumos_clip_score_b16.npy"
cls_data = np.load(ext_path)
thu_label_id = np.array(list(thumos_class.keys())) - 1  # get thumos class id



@DETECTORS.register_module()
class ZSDefDETR(DeformableDETR):
    def __init__(
        self,
        projection,
        transformer,
        neck=None,
        backbone=None,
        fe_estimator=None,
    ):
        super(ZSDefDETR, self).__init__(
            projection=projection,
            transformer=transformer,
            neck=neck,
            backbone=backbone,
        )
        self.fe_estimator = build_forground_estimator(fe_estimator)
        self.with_act_reg = transformer.with_act_reg

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        # forground estimator
        fe_loss_dict = self.fe_estimator(inputs, masks, kwargs['text_proto'], kwargs['img_feats'], kwargs['img_masks'], gt_labels)
        losses.update(fe_loss_dict)

        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        # padding masks is the opposite of valid masks
        if isinstance(masks, list):
            padding_masks = [~mask for mask in masks]
        elif isinstance(masks, torch.Tensor):
            padding_masks = ~masks
        else:
            raise TypeError("masks should be either list or torch.Tensor")

        transformer_loss = self.transformer.forward_train(
            x,
            padding_masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        losses.update(transformer_loss)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        vid_scores = self.fe_estimator(inputs, masks, kwargs['text_proto'], kwargs['img_feats'], kwargs['img_masks'])

        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        # padding masks is the opposite of valid masks
        if isinstance(masks, list):
            padding_masks = [~mask for mask in masks]
        elif isinstance(masks, torch.Tensor):
            padding_masks = ~masks
        else:
            raise TypeError("masks should be either list or torch.Tensor")

        output = self.transformer.forward_test(x, padding_masks, **kwargs)

        # video scores
        # pred_num = output["pred_logits"].shape[1]
        # vid_scores = vid_scores.unsqueeze(1).expand(-1, pred_num, -1)
        output["pred_video_scores"] = vid_scores

        predictions = output, masks[0]
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        output, masks = predictions
        pred_logits = output["pred_logits"]  #  [B,K,num_classes], before sigmoid
        pred_boxes = output["pred_boxes"]  # [B,K,2]
        video_scores = output["pred_video_scores"]

        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 200)
        bs, _, num_classes = pred_logits.shape

        # Select top-k confidence boxes for inference
        if self.with_act_reg:
            prob = pred_logits.sigmoid() * output["pred_actionness"]
        else:
            prob = pred_logits.sigmoid()

        pre_nms_topk = min(pre_nms_topk, prob.view(bs, -1).shape[1])
        topk_values, topk_indexes = torch.topk(prob.view(bs, -1), pre_nms_topk, dim=1)
        batch_scores = topk_values
        topk_boxes = torch.div(topk_indexes, num_classes, rounding_mode="floor")
        batch_labels = torch.fmod(topk_indexes, num_classes)

        batch_proposals = proposal_cw_to_se(pred_boxes) * masks.shape[-1]  # cw -> sw, 0~tscale
        batch_proposals = torch.gather(batch_proposals, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = batch_proposals[i].detach().cpu()  # [N,2]
            scores = batch_scores[i].detach().cpu()  # [N,class]
            labels = batch_labels[i].detach().cpu()  # [N]
            video_score = video_scores[i].detach().cpu()  # [true_class_num]

            # if not sliding window, do nms
            if post_cfg.sliding_window == False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                # labels = [ext_cls[label.item()] for label in labels]
                segments, labels, scores = self._video_class_inject(video_id, segments, scores, video_score, ext_cls)
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    def _video_class_inject(self, video_id, segments, scores, video_score, ext_cls):
        # >> 不使用外部分类器
        # sorted_scores, sorted_indices = torch.sort(video_score, descending=True)
        # sorted_ext_cls = [ext_cls[idx.item()] for idx in sorted_indices]
        # sorted_scores = sorted_scores.cpu().detach().numpy().tolist()

        # score_scale = 0.5
        # if len(ext_cls) > 10:
        #     score_scale = 0.1

        # selected_scores = sorted_scores[:2]
        # selected_class_names = sorted_ext_cls[:2]
        # selected_scores[-1] = selected_scores[-1] * score_scale
        # vid_cls_dict = dict(zip(selected_class_names, selected_scores))

        # >> 使用外部分类器
        if len(ext_cls) > 10:
            cuhk_score = np.array(cuhk_data_score[video_id])
            cuhk_classes = cuhk_data_action[np.argsort(-cuhk_score)]
            cuhk_score = cuhk_score[np.argsort(-cuhk_score)]

            cuhk_classes = cuhk_classes.tolist()
            cuhk_score = cuhk_score.tolist()
            new_cuhk_classes = []
            new_cuhk_score = []
            for i in range(len(cuhk_classes)):
                if cuhk_classes[i] in ext_cls:
                    new_cuhk_classes.append(cuhk_classes[i])
                    if len(new_cuhk_score) > 0:
                        new_cuhk_score.append(cuhk_score[i] * 0.1)
                    else:
                        new_cuhk_score.append(cuhk_score[i])
                if len(new_cuhk_classes) >= 2:
                    break
        else:
            video_cls = cls_data[int(video_id[-4:]) - 1][thu_label_id]  # order by video list, output 20
            video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
            unet_classes = [thu_label_id[video_cls_rank[-k - 1][1]] + 1 for k in range(20)]
            new_unet_classes = []
            for unet_class in unet_classes:
                new_unet_classes.append(thumos_class[unet_class])
            unet_classes = new_unet_classes
            unet_scores = [video_cls_rank[-k - 1][0] for k in range(20)]
            new_cuhk_classes = []
            new_cuhk_score = []
            for i in range(len(unet_classes)):
                if unet_classes[i] in ext_cls:
                    new_cuhk_classes.append(unet_classes[i])
                    if len(new_cuhk_score) > 0:
                        new_cuhk_score.append(unet_scores[i] * 0.5)
                    else:
                        new_cuhk_score.append(unet_scores[i])
                if len(new_cuhk_classes) >= 2:
                    break
        vid_cls_dict = dict(zip(new_cuhk_classes, new_cuhk_score))

        sorted_vid_cls_dict = dict(sorted(vid_cls_dict.items(), key=lambda item: item[1], reverse=True))

        new_segments = []
        new_labels = []
        new_scores = []
        # for segment, score in zip(segments, scores):
        for k, v in sorted_vid_cls_dict.items():
            new_segments.append(segments)
            new_labels.extend([k] * len(segments))
            new_scores.append(scores * v)

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores



    def get_optim_groups(self, cfg):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        lr_linear_proj_names = ["reference_points", "sampling_offsets"]

        param_dicts = [
            # non-backbone, non-offset
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("backbone")
                    and not match_name_keywords(n, lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": cfg.lr,
                "initial_lr": cfg.lr,
            },
            # offset
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("backbone") and match_name_keywords(n, lr_linear_proj_names) and p.requires_grad
                ],
                "lr": cfg.lr * 0.1,
                "initial_lr": cfg.lr * 0.1,
            },
        ]
        return param_dicts

    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, **kwargs)

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, **kwargs):
        # step1: inference the model
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg, **kwargs)

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results
