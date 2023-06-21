# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='HSencoder',
        image_size=256,
        frame_size=96,
        patch_size=16,
        stride=128,
        num_classes=1000,
        dim=512,
        depth=6,
        num_heads=16,
        mlp_dim=1024,
        cls_head_dim=768,
        dropout=0.1,
        emb_dropout=0.1,
        comb_strate='plain'),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=174,
        in_channels=768),
    test_cfg=dict(average_clips='prob'))


