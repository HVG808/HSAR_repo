_base_ = [
    '../../../_base_/models/HSAR/HSAR_linear_tiny.py',
    '../../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/sthv2/videos_train'
data_root_val = 'data/sthv2/videos_train'
ann_file_train = 'data/sthv2/sthv2_train_list_videos.txt'
ann_file_val = 'data/sthv2/sthv2_val_list_videos.txt'
ann_file_test = 'data/sthv2/sthv2_val_list_videos.txt'

# gpu_ids = [0]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        # frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=6,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,  # this lr is used for 1 gpu
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
auto_scale_lr = dict(enable=True, base_batch_size=32)
lr_config = dict(policy='step', step=[6, 13, 15, 17])
total_epochs = 18


# runtime settings
work_dir = 'OUTPUT/K400/HSAR_Base_sthv2'
find_unused_parameters = True

model = dict(backbone=dict(TMT_ckpt='saves/k400_16_16.pth',
                           vit_ckpt='saves/checkpoint_teacher.pth',
                           temporal='TMT',
                           dim=2048,
                           frame_size=64,
                           comb_strate='fusion',
                           Apool='cls',
                           Vpool='cls',
                           Fpool='cls',
                           dropout=0.2,
                           emb_dropout=0.2,
                           drop_path=0.2,
                           num_heads=16,
                           depth=6,
                           fusion_layer=6,
                           stride=64,
                           freeze_spa=True,
                           skip_mo=4
                           ),
             cls_head=dict(
                 type='HSARHead',
                 in_channels=2048,
                 num_classes=174,
             ))
