backbone_name_to_buildfunc = {
    "build_resnet_fpn_backbone_resnet": ["senet", "augmix", "fbnet",
                                         "resnext_swsl", "ssl_res", "resnet18"],
    "build_resnet_fpn_backbone_mobilenet": ["mobilenet_v2"],
    "build_convnext_fpn_backbone": ["convnext_base_1k", "convnext_base_21k", "convnext_base_21k_senet"],
    "build_resnet_fpn_backbone_efficient": ["efficientnet_b2", "tf_efficientnet_b0_ns", "tf_efficientnet_b2_ns"],
    "build_resnet_fpn_backbone_caffe": ["caffe_res50", "caffe_res101"],
    "build_fpn_backbone_selike_resnet": ["fbnet_senet"],
}
backbone_name_to_pixel_stats = {
    "[[103.530, 116.280, 123.675], [1, 1, 1]]": ["caffe_res50", "caffe_res101"],
    "[[123.675, 116.280, 103.530], [58.395, 57.120, 57.375]]": ["fbnet", "fbnet_senet", "augmix",
                                                                "resnext_swsl", "ssl_res",
                                                                "senet", "senet_analyze", 'resnet18',
                                                                "fbnet_senet", "mobilenet_v2",
                                                                "convnext_base_1k", "convnext_base_21k",
                                                                "convnext_base_21k_senet",
                                                                "efficientnet_b2", "tf_efficientnet_b0_ns",
                                                                "tf_efficientnet_b2_ns"],
}
backbone_name_to_buildfunc = {name: k for k, v in backbone_name_to_buildfunc.items() for name in v}
backbone_name_to_pixel_stats = {name: k for k, v in backbone_name_to_pixel_stats.items() for name in v}


def map_backbone2model(cfg, args):
    MODEL_NAME = cfg.MODEL.PRETRAIN_NAME
    if cfg.TRAIN.SEPRETRAIN:
        MODEL_NAME += "_senet"
    cfg.MODEL.BACKBONE.NAME = backbone_name_to_buildfunc[MODEL_NAME]
    cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD = eval(backbone_name_to_pixel_stats[MODEL_NAME])
    # For stability of training, we use GN for convnext.
    if 'convnext' in MODEL_NAME:
        cfg.MODEL.FPN.NORM = "GN"
    else:
        cfg.MODEL.FPN.NORM = ""
    if args.eval_only:
        if cfg.MODEL.PRETRAIN_NAME == 'caffe_res101':
            cfg.MODEL.RESNETS.DEPTH = 101
        return
    if cfg.MODEL.PRETRAIN_NAME == 'caffe_res50':
        if cfg.MODEL.WEIGHTS == "":
            cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    elif cfg.MODEL.PRETRAIN_NAME == 'caffe_res101':
        if cfg.MODEL.WEIGHTS == "":
            cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
        cfg.MODEL.RESNETS.DEPTH = 101
