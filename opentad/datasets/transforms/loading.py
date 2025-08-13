import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np
import json

from ..builder import PIPELINES
from torch.nn import functional as F


@PIPELINES.register_module()
class LoadAnetVideoLevelLabels:
    def __init__(self, vll_path, test_class_map, all_class_map):
        with open(vll_path, "r") as f:
            cuhk_data = json.load(f)
        self.cuhk_data_score = cuhk_data["results"]
        self.cuhk_data_action = np.array(cuhk_data["class"])

        self.topk = 2
        self.ext = []
        with open(test_class_map, 'r', encoding='utf-8') as file:
            self.ext = [line.strip() for line in file.readlines()]

        with open(all_class_map, "r", encoding="utf8") as f:
            lines = f.readlines()
        self.class_map = [item.rstrip("\n") for item in lines]

    def _get_vll_labels(self, video_name):
        # sort video classification
        cuhk_score = np.array(self.cuhk_data_score[video_name])
        cuhk_classes = self.cuhk_data_action[np.argsort(-cuhk_score)]
        cuhk_score = cuhk_score[np.argsort(-cuhk_score)]

        vll_classes, vll_scores = [], []
        need_k = 0
        for k in range(len(cuhk_classes)):
            if cuhk_classes[k] in self.ext:
                vll_classes.append(cuhk_classes[k])
                if need_k == 0:
                    vll_scores.append(cuhk_score[k])
                else:
                    vll_scores.append(cuhk_score[k] * 0.5)
                need_k += 1

                if need_k >= self.topk:
                    break

        new_vll_classes = []
        for vll_c in vll_classes:
            new_vll_classes.append(self.class_map.index(vll_c))

        return new_vll_classes, vll_scores

    def __call__(self, results):
        video_name = results["video_name"]

        vll_classes, vll_scores = self._get_vll_labels(video_name)

        results["vll_classes"] = vll_classes
        results['vll_scores'] = vll_scores
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"tbd..."
        return repr_str


@PIPELINES.register_module()
class LoadThumosVideoLevelLabels:
    def __init__(self, vll_path, test_class_map, all_class_map):
        self.thumos_class = {
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

        self.cls_data = np.load(vll_path)
        self.thu_label_id = np.array(list(self.thumos_class.keys())) - 1  # get thumos class id
        self.topk = 2
        self.ext = []
        with open(test_class_map, 'r', encoding='utf-8') as file:
            self.ext = [line.strip() for line in file.readlines()]

        with open(all_class_map, "r", encoding="utf8") as f:
            lines = f.readlines()
        self.class_map = [item.rstrip("\n") for item in lines]

    def _get_vll_labels(self, video_name):
        # sort video classification
        video_cls = self.cls_data[int(video_name[-4:]) - 1][self.thu_label_id]  # order by video list, output 20
        video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
        vll_classes, vll_scores = [], []
        need_k = 0
        for k in range(20):
            if self.thumos_class[self.thu_label_id[video_cls_rank[-k - 1][1]] + 1] in self.ext:
                vll_classes.append(self.thumos_class[int(self.thu_label_id[video_cls_rank[-k - 1][1]] + 1)])
                if need_k == 0:
                    vll_scores.append(video_cls_rank[-k - 1][0])
                else:
                    vll_scores.append(video_cls_rank[-k - 1][0] * 0.5)
                need_k += 1
            if need_k >= self.topk:
                break

        new_vll_classes = []
        for vll_c in vll_classes:
            new_vll_classes.append(self.class_map.index(vll_c))

        return new_vll_classes, vll_scores

    def __call__(self, results):
        video_name = results["video_name"]

        vll_classes, vll_scores = self._get_vll_labels(video_name)

        results["vll_classes"] = vll_classes
        results['vll_scores'] = vll_scores
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"tbd..."
        return repr_str

@PIPELINES.register_module()
class LoadCLIPFeats:
    def __init__(self, text_feat_path, image_feat_dir, all_class_map, prefix="", suffix=""):
        self.text_feat_path = text_feat_path
        self.image_feat_dir = image_feat_dir
        self.prefix = prefix
        self.suffix = suffix

        with open(all_class_map, "r", encoding="utf8") as f:
            lines = f.readlines()
        self.class_map = [item.rstrip("\n") for item in lines]

        self.text_feats_dict = self._get_text_feat()

    def _get_text_feat(self):
        text_feat_dict = np.load(self.text_feat_path, allow_pickle=True)
        text_feat_dict = dict(text_feat_dict)

        text_feats = []
        for _cls_name in self.class_map:
            text_feat = text_feat_dict[_cls_name]
            text_feats.append(text_feat)
        all_text_feats = np.concatenate(text_feats, axis=0)

        return all_text_feats

    def _get_image_feat(self, video_name):
        image_feat_path = os.path.join(self.image_feat_dir, f"{video_name}.npy")
        if not os.path.exists(image_feat_path):
            image_feat_path = os.path.join(self.image_feat_dir, f"v_{video_name}.npy")
        image_feats = np.load(image_feat_path)
        return image_feats

    def __call__(self, results):
        video_name = results["video_name"]

        image_feats = self._get_image_feat(video_name)

        results["clip_text_feat"] = self.text_feats_dict.astype(np.float32)
        results['clip_image_feat'] = image_feats.astype(np.float32)
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"tbd..."
        return repr_str


@PIPELINES.register_module()
class LoadTextFeats:
    def __init__(self, text_feat_path, ann_file, class_map, prefix="", suffix=""):
        self.text_feat_path = text_feat_path
        self.ann_file = ann_file
        self.class_map = self.get_class_map(class_map)
        self.prefix = prefix
        self.suffix = suffix

        self.text_feats = self._get_text_proto()

    def get_class_index(self, gt_json_path, class_map_path):
        with open(gt_json_path, "r") as f:
            anno = json.load(f)

        anno = anno["database"]
        class_map = []
        for video_name in anno.keys():
            if "annotations" in anno[video_name]:
                for tmpp_data in anno[video_name]["annotations"]:
                    if tmpp_data["label"] not in class_map:
                        class_map.append(tmpp_data["label"])

        class_map.sort()
        f2 = open(class_map_path, "w")
        for name in class_map:
            f2.write(name + "\n")
        f2.close()
        return class_map

    def get_class_map(self, class_map_path):
        if not os.path.exists(class_map_path):
            class_map = self.get_class_index(self.ann_file, class_map_path)
        else:
            with open(class_map_path, "r", encoding="utf8") as f:
                lines = f.readlines()
            class_map = [item.rstrip("\n") for item in lines]
        return class_map

    def _get_text_proto(self):
        text_proto_dict = np.load(self.text_feat_path, allow_pickle=True)
        text_proto_dict = dict(text_proto_dict)
        text_proto = []
        for _cls_name in self.class_map:
            text_feat = text_proto_dict[_cls_name]
            text_proto.append(text_feat)
        text_proto = np.concatenate(text_proto, axis=0)
        return text_proto

    def __call__(self, results):
        results["text_proto"] = self.text_feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"tbd..."
        return repr_str

@PIPELINES.register_module()
class LoadFeats:
    def __init__(self, feat_format, prefix="", suffix=""):
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        # check feat format
        if isinstance(self.feat_format, str):
            self.check_feat_format(self.feat_format)
        elif isinstance(self.feat_format, list):
            for feat_format in self.feat_format:
                self.check_feat_format(feat_format)

    def check_feat_format(self, feat_format):
        assert feat_format in ["npy", "npz", "pt", "csv", "pkl"], print(f"not support {feat_format}")

    def read_from_tensor(self, file_path):
        feats = torch.load(file_path).float()
        return feats

    def read_from_npy(self, file_path):
        feats = np.load(file_path).astype(np.float32)
        return feats

    def read_from_npz(self, file_path):
        feats = np.load(file_path)["feats"].astype(np.float32)
        return feats

    def read_from_csv(self, file_path):
        feats = pd.read_csv(file_path, dtype="float32").to_numpy()
        feats = feats.astype(np.float32)
        return feats

    def read_from_pkl(self, file_path):
        feats = pickle.load(open(file_path, "rb"))
        feats = feats.astype(np.float32)
        return feats

    def load_single_feat(self, file_path, feat_format):
        try:
            if feat_format == "npy":
                feats = self.read_from_npy(file_path)
            elif feat_format == "npz":
                feats = self.read_from_npz(file_path)
            elif feat_format == "pt":
                feats = self.read_from_tensor(file_path)
            elif feat_format == "csv":
                feats = self.read_from_csv(file_path)
            elif feat_format == "pkl":
                feats = self.read_from_pkl(file_path)
        except:
            print("Missing data:", file_path)
            exit()
        return feats

    def __call__(self, results):
        video_name = results["video_name"]

        if isinstance(results["data_path"], str):
            file_path = os.path.join(results["data_path"], f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}")
            feats = self.load_single_feat(file_path, self.feat_format)
        elif isinstance(results["data_path"], list):
            feats = []

            # check if the feat_format is a list
            if isinstance(self.feat_format, str):
                self.feat_format = [self.feat_format] * len(results["data_path"])

            for data_path, feat_format in zip(results["data_path"], self.feat_format):
                file_path = os.path.join(data_path, f"{self.prefix}{video_name}{self.suffix}.{feat_format}")
                feats.append(self.load_single_feat(file_path, feat_format))

            max_len = max([feat.shape[0] for feat in feats])
            for i in range(len(feats)):
                if feats[i].shape[0] != max_len:
                    # assume the first dimension is T
                    tmp_feat = F.interpolate(
                        torch.Tensor(feats[i]).permute(1, 0).unsqueeze(0),
                        size=max_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                    feats[i] = tmp_feat.permute(1, 0).numpy()
            feats = np.concatenate(feats, axis=1)

        # sample the feature
        sample_stride = results.get("sample_stride", 1)
        if sample_stride > 1:
            feats = feats[::sample_stride]

        results["feats"] = feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"feat_format={self.feat_format}"
        return repr_str


@PIPELINES.register_module()
class SlidingWindowTrunc:
    """This is used for sliding window dataset, which will give a window start and window end in the result dict,
    and we will extract the window features, also pad to fixed length"""

    def __init__(self, with_mask=True):
        self.with_mask = with_mask

    def __call__(self, results):
        assert "window_size" in results.keys(), "should have window_size as a key"
        assert isinstance(results["feats"], torch.Tensor)
        window_size = results["window_size"]

        feats_length = results["feats"].shape[0]
        start_idx = min(results["feature_start_idx"], feats_length)
        end_idx = min(results["feature_end_idx"] + 1, feats_length)

        window_feats = results["feats"][start_idx:end_idx]
        valid_len = window_feats.shape[0]

        # if the valid window is smaller than window size, pad with -1
        if valid_len < window_size:
            pad_data = torch.zeros(window_size - valid_len, window_feats.shape[1])
            window_feats = torch.cat((window_feats, pad_data), dim=0)

        # if we need padding mask (valid is 1, pad is 0)
        if self.with_mask:
            if valid_len < window_size:
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)])
            else:
                masks = torch.ones(window_size)
            results["masks"] = masks.bool()

        results["feats"] = window_feats.float()
        return results


@PIPELINES.register_module()
class LoadImageFeats:
    """This is used for sliding window dataset, which will give a window start and window end in the result dict,
    and we will extract the window features, also pad to fixed length"""

    def __init__(self, text_data_path, data_fps, feat_format, prefix="", suffix="", with_mask=True):
        self.text_data_path = text_data_path
        self.data_fps = data_fps
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        self.with_mask = with_mask

    def read_from_npy(self, file_path):
        feats = np.load(file_path).astype(np.float32)
        return feats

    def load_single_feat(self, file_path, feat_format):
        try:
            if feat_format == "npy":
                feats = self.read_from_npy(file_path)
        except:
            print("Missing data:", file_path)
            exit()
        return feats

    def __call__(self, results):
        video_name = results["video_name"]

        file_path = os.path.join(self.text_data_path, f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}")
        text_feats = self.load_single_feat(file_path, self.feat_format)
        text_feats = torch.from_numpy(text_feats)

        window_size = results.get("window_size", -1)
        if window_size > 0:
            # >> thumos14 sliding window
            img_feat_window_size = 1000
            snippet_stride = results['snippet_stride']
            offset_frames = results['offset_frames']
            fps = results['fps']

            feats_length = text_feats.shape[0]
            start_idx = min((results["feature_start_idx"] * snippet_stride + offset_frames) / fps * self.data_fps, feats_length)
            end_idx = min((results["feature_end_idx"] * snippet_stride + offset_frames) / fps * self.data_fps, feats_length)
            start_idx = int(start_idx)
            end_idx = int(end_idx)

            window_feats = text_feats[start_idx:end_idx]
            valid_len = window_feats.shape[0]

            # if the valid window is smaller than window size, pad with -1
            if valid_len < img_feat_window_size:
                pad_data = torch.zeros(img_feat_window_size - valid_len, window_feats.shape[1])
                window_feats = torch.cat((window_feats, pad_data), dim=0)

            # if we need padding mask (valid is 1, pad is 0)
            if self.with_mask:
                if valid_len < img_feat_window_size:
                    masks = torch.cat([torch.ones(valid_len), torch.zeros(img_feat_window_size - valid_len)])
                else:
                    masks = torch.ones(img_feat_window_size)
                results["img_masks"] = masks.bool()

            results["img_feats"] = window_feats.float()
        else:
            # >> activitynet-1.3 resize
            tscale = results["resize_length"]
            text_feats = text_feats.permute(1, 0)
            img_feats = F.interpolate(text_feats.unsqueeze(0), size=tscale, mode="nearest").squeeze(0)
            img_feats = img_feats.permute(1, 0)
            masks = torch.ones(img_feats.shape[0])
            results["img_masks"] = masks.bool()
            results["img_feats"] = img_feats.float()

        return results


@PIPELINES.register_module()
class RandomTrunc:
    """Crops features within a window such that they have a large overlap with ground truth segments.
    Withing the cropping ratio, the length is sampled."""

    def __init__(
        self,
        trunc_len,
        trunc_thresh,
        crop_ratio=None,
        max_num_trials=200,
        has_action=True,
        no_trunc=False,
        pad_value=0,
        channel_first=False,
    ):
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.max_num_trials = max_num_trials
        self.has_action = has_action
        self.no_trunc = no_trunc
        self.pad_value = pad_value
        self.channel_first = channel_first

    def trunc_features(self, feats, gt_segments, gt_labels, gt_real_labels, offset, clip_image_feat, snippet_stride, offset_frames, fps, resize_length, duration):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = self.trunc_len
        if feat_len <= self.trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels, gt_real_labels, clip_image_feat
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels, gt_real_labels, clip_image_feat

        # try a few times till a valid truncation with at least one action
        for _ in range(self.max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = torch.as_tensor([st, ed], dtype=torch.float32)

            # compute the intersection between the sampled window and all segments
            window = window[None].repeat(num_segs, 1)
            left = torch.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = torch.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = (right - left).clamp(min=0)
            area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            if self.no_trunc:
                # with at least one action and not truncating any actions
                seg_trunc_idx = (inter_ratio > 0.0) & (inter_ratio < 1.0)
                if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                    break
            elif self.has_action:
                # with at least one action
                if seg_idx.sum().item() > 0:
                    break
            else:
                # without any constraints
                break

        feats = feats[st:ed, :]  # [T,C]
        gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        gt_real_labels = gt_real_labels[seg_idx]  # [N]

        if fps != -1:
            clip_st = int((st * snippet_stride + offset_frames) / fps * 10)
            clip_ed = int((ed * snippet_stride + offset_frames) / fps * 10)
            clip_st = max(0, clip_st)
            clip_ed = min(clip_image_feat.shape[0], clip_ed)
            clip_image_feat = clip_image_feat[clip_st:clip_ed, :]
        else:
            clip_st = int((st / resize_length * duration) * 10)
            clip_ed = int((ed / resize_length * duration) * 10)
            clip_st = max(0, clip_st)
            clip_ed = min(clip_image_feat.shape[0], clip_ed)
            clip_image_feat = clip_image_feat[clip_st:clip_ed, :]

        return feats, gt_segments, gt_labels, gt_real_labels, clip_image_feat

    def pad_features(self, feats):
        feat_len = feats.shape[0]
        if feat_len < self.trunc_len:
            feats_pad = torch.ones((self.trunc_len - feat_len,) + feats.shape[1:]) * self.pad_value
            feats = torch.cat([feats, feats_pad], dim=0)
            masks = torch.cat([torch.ones(feat_len), torch.zeros(self.trunc_len - feat_len)])
            return feats, masks
        else:
            return feats, torch.ones(feat_len)

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        offset = 0

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [C,T] -> [T,C]

        snippet_stride = results.get("snippet_stride", None)
        offset_frames = results.get("offset_frames", None)
        fps = results.get("fps", None)
        resize_length = results.get("resize_length", None)
        duration = results.get("duration", None)

        # truncate the features
        feats, gt_segments, gt_labels, gt_real_labels, clip_image_feat = self.trunc_features(
            results["feats"],
            results["gt_segments"],
            results["gt_labels"],
            results["gt_real_labels"],
            offset,
            results["clip_image_feat"],
            snippet_stride,
            offset_frames,
            fps,
            resize_length,
            duration
        )

        # pad the features to the fixed length
        feats, masks = self.pad_features(feats)

        results["feats"] = feats.float()
        results["masks"] = masks.bool()
        results["gt_segments"] = gt_segments
        results["gt_labels"] = gt_labels
        results["gt_real_labels"] = gt_real_labels
        results["clip_image_feat"] = clip_image_feat

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [T,C] -> [C,T]
        return results
