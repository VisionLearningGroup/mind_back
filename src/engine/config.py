# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_gdet_config(cfg):
    """
    Add config for Our training
    """
    # Input keyword for pre-trained model. e.g., in1k, augmix, fbnet...
    cfg.MODEL.PRETRAIN_NAME = 'fbnet'
    cfg.MODEL.FPN.NAME = "FPN_RES"
    cfg.MODEL.FPN.NAS_STACK = 7  # 7 is default, but increase memory a lot
    cfg.SOLVER.COEFF = 0.1
    cfg.SOLVER.EMA_REG = False
    cfg.TRAIN = CN()
    cfg.TRAIN.USE_AUGMIX = False
    # Use this flag will freeze backbone
    cfg.TRAIN.PRETRAIN = False
    cfg.TRAIN.ITER_GRAD_COMP = 5000
    cfg.TRAIN.AUGMIX_WIDTH = 1
    cfg.TRAIN.EMA_REG = False
    cfg.TRAIN.REG = CN()
    # Apply weight regularization based on detection loss.
    cfg.TRAIN.REG.RGN = False
    # Apply EWC based on detection loss.
    cfg.TRAIN.REG.EWC = False
    # Apply simple weight distance based regularization.
    cfg.TRAIN.REG.PLAIN_DIST = False
    # SE Module Pre-training
    cfg.TRAIN.SEPRETRAIN = False
    # TEST on corruptions
    cfg.TEST.NOISE_ALL = False
    # This flag is to resume training.
    # We set the reference weight as the initial backbone.
    cfg.MODEL.REF_WEIGHTS = None
