# MGCA-Net: Multi-Grained Category-Aware Network for Open-Vocabulary Temporal Action Localization

official code for **MGCA-Net: Multi-Grained Category-Aware Network for Open-Vocabulary Temporal Action Localization**

# Preprocess

## Environment

- Our code is built upon the codebase from [OpenTAD](https://github.com/sming256/OpenTAD), and we would like to express our gratitude for their outstanding work.


## Data Process

### Get thumos-14 dataset

Please follow the data processing of OpenTAD to obtain the THUMOS'14 dataset.

Add the obtained thumos-14 dataset to the data directory. The directory structure is as follows:

── data \
   ├── ov_data \
   └── thumos-14

### Extract image and text features

- Use `data/ov_data/data_preprocess/get_image_clip_feats.py` and `data/ov_data/data_preprocess/get_text_clip_feats.py` to extract image and text features.

- Alternatively, you can use our pre-extracted features.
  - image features: [BaiduDisk](https://pan.baidu.com/s/189U4u1pu4HhnWPSeDAXBGQ?pwd=qvss) (code: qvss)
  - text features: `data/ov_data/clip_scores/thumos_clip_score_b16_multi.npy`


# Train MGCA-Net

- Run the `train.sh` script for training

# Training logs and checkpoints

- [BaiduDisk](https://pan.baidu.com/s/1J0TP4UPAbqIKs6tIqIahkA?pwd=auxa) (code: auxa)

# Results

| <center>Split</center> | 0.3  | 0.4  | 0.5  | 0.6  | 0.7  | Avg. |
| ---------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 75% Seen - 25% Unseen  | 66.4 | 59.7 | 49.6 | 38.2 | 26.2 | 48.0 |
| 50% Seen - 50% Unseen  | 58.0 | 51.6 | 42.6 | 33.0 | 21.5 | 41.3 |
