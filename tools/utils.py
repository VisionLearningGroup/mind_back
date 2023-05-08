import os
from src.engine import add_gdet_config
from map_backbone import map_backbone2model
import torch
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch
import detectron2.utils.comm as comm


def setup(args, eval_only=False):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_gdet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    train_mode = name_trainmode(cfg)
    map_backbone2model(cfg, args)
    if not eval_only and not args.resume:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, train_mode)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def set_up_logger(output_dir):
    rank = comm.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank, name="gdet_training")
    return logger


def name_trainmode(cfg):
    train_mode = ""
    if cfg.TRAIN.REG.PLAIN_DIST:
        train_mode += "_Plain_Distance_REG"
    elif cfg.TRAIN.REG.RGN:
        train_mode += "_RGN_WEIGHTED_REG"
    elif cfg.TRAIN.REG.EWC:
        train_mode += "_EWC_REG"
    if len(train_mode):
        train_mode += "_network_{}_coeff_{}".format(cfg.MODEL.PRETRAIN_NAME, cfg.SOLVER.COEFF)
    else:
        train_mode += "_network_{}".format(cfg.MODEL.PRETRAIN_NAME)
        if cfg.TRAIN.PRETRAIN:
            train_mode += "_pretrain"
        elif cfg.TRAIN.SEPRETRAIN:
            train_mode += "_SE-pretrain"
    train_mode = "_".join(cfg.DATASETS.TRAIN) + "_" + train_mode
    return train_mode


def get_gradscaler(cfg, model, dataloader, iter_precomp, logger):
    scale_param_task = {}
    for data, iteration in zip(dataloader, range(iter_precomp)):
        if iteration > 0 and (
                (iteration + 1) % 20 == 0 or iteration == iter_precomp - 1
        ):
            logger.info("Weight Importance computation Iter {} finished".format(iteration))
        model.zero_grad()
        loss_dict = model(data)
        losses = sum(loss_dict.values())
        losses.backward()
        for name, p_tgt in model.named_parameters():
            if 'bottom_up' in name:
                if p_tgt.grad is not None:
                    norm_grad = torch.sqrt(p_tgt.grad * p_tgt.grad)
                    if cfg.TRAIN.REG.EWC:
                        scale_param_task[name] = scale_param_task.get(name, 0) + norm_grad / iter_precomp
                        continue
                    norm_param = torch.sqrt(p_tgt * p_tgt).detach()
                    if len(norm_grad.shape) == 4:
                        norm_grad = norm_grad.sum(-1).sum(-1)
                        norm_param = norm_param.sum(-1).sum(-1)
                    scale_param_task[name] = scale_param_task.get(name, 0) + (
                            norm_grad / (norm_param + 1e-8)) / iter_precomp
        del losses
    return scale_param_task


def gradient_modifier(cfg, model, model_ref, scale_param_task, lr):
    for (name, p_tgt), (_, p_src) in zip(model.named_parameters(), model_ref.named_parameters()):
        if 'bottom_up' in name and 'se_module' not in name:
            if p_tgt.grad is not None:
                if cfg.TRAIN.REG.PLAIN_DIST:
                    scaling = cfg.SOLVER.COEFF
                else:
                    value = scale_param_task[name]
                    scale_unclip = value
                    if not cfg.TRAIN.REG.EWC:
                        if len(p_src.shape) == 4:
                            h, w, _, _ = p_src.shape
                            scale_unclip = value.view(h, w, 1, 1)
                    scaling = torch.clip(cfg.SOLVER.COEFF * scale_unclip, min=0, max=1. / lr)
                p_tgt.grad += (p_tgt - p_src) * scaling
