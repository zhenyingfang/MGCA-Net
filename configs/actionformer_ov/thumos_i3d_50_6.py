_base_ = [
    "../_base_/models/actionformer.py",  # model config
]

dataset_type = "ThumosPaddingDataset"
annotation_path = "data/ov_data/zs_data/split50/THUMOS14/split_6.json"
class_map = "data/ov_data/zs_data/split50/THUMOS14/category_idx_6.txt"
test_class_map = "data/ov_data/zs_data/split50/THUMOS14/category_idx_6_test.txt"
data_path = "data/thumos-14/features/i3d_actionformer_stride4_thumos/"
block_list = data_path + "missing_files.txt"

text_feat_path = "data/ov_data/data_preprocess/clip_text_feats/thumos_text_proto_b16_t0.npz"
image_feat_dir = "data/thumos-14/clip_img_feat"
all_class_map = "data/thumos-14/annotations/category_idx.txt"

trunc_len = 2304

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        offset_frames=8,
        class_agnostic=True,
        real_class_map=all_class_map,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="LoadCLIPFeats", text_feat_path=text_feat_path, image_feat_dir=image_feat_dir, all_class_map=all_class_map),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels", "gt_real_labels", "clip_text_feat", "clip_image_feat", "class_map_idx"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels", "gt_real_labels", "clip_text_feat", "clip_image_feat", "class_map_idx"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=test_class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        offset_frames=8,
        class_agnostic=True,
        real_class_map=all_class_map,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="LoadCLIPFeats", text_feat_path=text_feat_path, image_feat_dir=image_feat_dir, all_class_map=all_class_map),
            dict(type="LoadThumosVideoLevelLabels", vll_path="data/ov_data/clip_scores/thumos_clip_score_b16_multi.npy", test_class_map=test_class_map, all_class_map=all_class_map),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels", "clip_text_feat", "clip_image_feat"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels", "clip_text_feat", "clip_image_feat", "vll_classes", "vll_scores"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=test_class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        offset_frames=8,
        class_agnostic=True,
        real_class_map=all_class_map,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="LoadCLIPFeats", text_feat_path=text_feat_path, image_feat_dir=image_feat_dir, all_class_map=all_class_map),
            dict(type="LoadThumosVideoLevelLabels", vll_path="data/ov_data/clip_scores/thumos_clip_score_b16_multi.npy", test_class_map=test_class_map, all_class_map=all_class_map),
            dict(type="ConvertToTensor", keys=["feats", "clip_text_feat", "clip_image_feat"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "clip_text_feat", "clip_image_feat", "vll_classes", "vll_scores"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

model = dict(
    type="ActionFormer",
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=2048,
        out_channels=512,
        arch=(2, 2, 5),  # layers in embed / stem / branch
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=19),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=2304,
        input_pdrop=0.2,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        neg_num=3,
        score_thres=0.1,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)


solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=35)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        iou_threshold=0.1,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    external_cls=dict(
        type="UntrimmedNetTHUMOSClassifier",
        path="data/ov_data/clip_scores/thumos_clip_score_b16_multi.npy",
        # path="data/thumos-14/classifiers/uNet_test.npy",
        topk=2,
        test_class_map=test_class_map,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=20,
)

work_dir = "exps/thumos/actionformer_i3d_50/6"
