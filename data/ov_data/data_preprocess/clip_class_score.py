import os
import json
import torch
import numpy as np
from tqdm import tqdm
import glob
import copy


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    ext_path = "data/thumos-14/classifiers/uNet_test.npy"
    cls_data = np.load(ext_path)
    thu_label_id = np.array(list(thumos_class.keys())) - 1  # get thumos class id

    new_cls_data = copy.deepcopy(cls_data)

    ori_gt_path = "data/thumos-14/annotations/thumos_14_anno.json"
    with open(ori_gt_path, "r") as f:
        ori_gt_data = json.load(f)
    ori_gt_data = ori_gt_data['database']

    vid_names = []
    for vid_name in list(ori_gt_data.keys()):
        if ori_gt_data[vid_name]['subset'] == 'validation':
            vid_names.append(vid_name)

    text_feat_paths = glob.glob(f"data/ov_data/data_preprocess/clip_text_feats/*.npz")
    for vid_name in vid_names:
        all_logits_per_image = []
        for text_feat_path in text_feat_paths:
            text_proto_dict = np.load(text_feat_path, allow_pickle=True)
            text_proto_dict = dict(text_proto_dict)

            text_proto = []
            for _cls_name in list(thumos_class.values()):
                text_feat = text_proto_dict[_cls_name]
                text_proto.append(text_feat)
            text_proto = np.concatenate(text_proto, axis=0)
            text_proto = torch.from_numpy(text_proto).to(device)

            img_feat_path = f"data/thumos-14/clip_img_feat/{vid_name}.npy"
            processed_images = np.load(img_feat_path)

            dtype = text_proto.dtype
            img_feats = torch.from_numpy(processed_images).to(device).to(dtype)
            logits_per_image = img_feats @ text_proto.t()
            T, C = logits_per_image.shape
            top_values, top_tmp = torch.topk(logits_per_image, T // 8, dim=0)
            result = top_values.mean(dim=0, keepdim=True)
            all_logits_per_image.append(result)

        stacked_logits = torch.stack(all_logits_per_image, dim=0)
        result = torch.mean(stacked_logits, dim=0)
        cls_score = result[0].cpu().detach().numpy().tolist()

        vid_idx = int(vid_name[-4:]) - 1
        new_cls_data[vid_idx][thu_label_id] = np.array(cls_score)

    dst_path = "data/ov_data/clip_scores/thumos_clip_score_b16_multi.npy"
    np.save(dst_path, new_cls_data)

    print(f"save to {dst_path}")
    print("Done!")
