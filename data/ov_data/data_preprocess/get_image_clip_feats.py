import os
import cv2
import torch
import json
import clip
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_clip_score(model, preprocess, device, vid_name, dst_path):
    video_path = f"data/thumos-14/raw_data/video/{vid_name}.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 10)

    frame_count = 0
    processed_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            processed_image = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(processed_image)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                processed_images.append(image_features)

        frame_count += 1

    cap.release()

    img_feats = torch.cat(processed_images, dim=0).float()
    img_feats_npy = img_feats.cpu().detach().numpy()
    np.save(dst_path, img_feats_npy)


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/16", device=device)
    for _, param in model.named_parameters():
        param.requires_grad = False

    ori_gt_path = "data/thumos-14/annotations/thumos_14_anno.json"
    with open(ori_gt_path, "r") as f:
        ori_gt_data = json.load(f)
    ori_gt_data = ori_gt_data['database']

    vid_names = []
    for vid_name in list(ori_gt_data.keys()):
        vid_names.append(vid_name)
    
    dst_dir = "data/thumos-14/clip_img_feat"
    os.makedirs(dst_dir, exist_ok=True)
    for vid_name in tqdm(vid_names):
        dst_path = os.path.join(dst_dir, f"{vid_name}.npy")
        get_clip_score(model, preprocess, device, vid_name, dst_path)

    print('Done!')
