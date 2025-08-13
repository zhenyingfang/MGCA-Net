import os
import numpy as np
import clip
import torch

multiple_templates = [
    "The action of {}.",
    "an action of {article} {} in the image.",
    "itaa of {article} {}.",
    "itaa of the {}.",
    "{}",
    "A human is performing the {} action.",
    "a photo of action {}.",
    "an image of action {}.",
    "a blurry photo of the {}.",
]


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)

    class_name_path = "data/ov_data/data_preprocess/thumos_name.txt"
    with open(class_name_path, 'r') as file:
        class_names = file.readlines()
        class_names = [line.strip() for line in class_names]
    
    text_proto_dict = dict()

    for temp_idx in range(len(multiple_templates)):
        for _cls_name in class_names:
            # text = clip.tokenize([f"The action of {_cls_name}"]).to(device)
            text = clip.tokenize([multiple_templates[temp_idx].format(processed_name(_cls_name, rm_dot=True), article=article(_cls_name))]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_proto_dict[_cls_name] = text_features.cpu().detach().numpy()
        save_path = f"data/ov_data/data_preprocess/clip_text_feats/thumos_text_proto_b16_t{temp_idx}.npz"
        np.savez(save_path, **text_proto_dict)
    
    print('Done!')
