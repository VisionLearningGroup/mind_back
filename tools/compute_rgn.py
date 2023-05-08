"""
Detectron2 training script with a copy-paste augmentation training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in LDET.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use LDET as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import json
import os

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    EventStorage,
)
from torch.nn.parallel import DistributedDataParallel

from src.data import DatasetMapper, build_detection_train_loader
from trainer import setup, set_up_logger


# logger = logging.getLogger("gdet_training")

def get_gradscaler(cfg, model, dataloader, iter_precomp, logger):
    scale_param_task = {}
    gradient_param = {}
    loss_mean = 0
    num_params = 0
    for name, p_tgt in model.named_parameters():
        if 'bottom_up' not in name:
            num_params += p_tgt.numel()
    print("num_params: ", num_params)
    for data, iteration in zip(dataloader, range(iter_precomp)):
        if iteration > 0 and (
                (iteration + 1) % 20 == 0 or iteration == iter_precomp - 1
        ):
            logger.info("Weight Importance computation Iter {} finished".format(iteration))
        model.zero_grad()
        loss_dict = model(data)
        losses = sum(loss_dict.values())
        losses.backward()
        loss_mean += losses.item()
        for name, p_tgt in model.named_parameters():
            if 'bottom_up' in name and 'se_module' not in name and 'se.' not in name:
                if p_tgt.grad is not None:
                    norm_grad = torch.sqrt(p_tgt.grad * p_tgt.grad)
                    if cfg.TRAIN.REG.EWC:
                        scale_param_task[name] = scale_param_task.get(name, 0) + norm_grad / iter_precomp
                        continue
                    norm_param = torch.sqrt(p_tgt * p_tgt).detach()
                    if len(norm_grad.shape) == 4:
                        norm_grad = norm_grad.sum(-1).sum(-1)
                        norm_param = norm_param.sum(-1).sum(-1)
                    gradient_param[name] = gradient_param.get(name, 0) + norm_grad / iter_precomp
                    scale_param_task[name] = scale_param_task.get(name, 0) + (
                            norm_grad / (norm_param + 1e-8)) / iter_precomp
        del losses
    scale_param_task = {k1: v1.mean().item() for k1, v1 in scale_param_task.items()}
    gradient_param = {k1: v1.mean().item() for k1, v1 in gradient_param.items()}
    print(scale_param_task)
    print('mean: ', sum([v for k, v in scale_param_task.items()]) / len(scale_param_task))
    print("loss_mean", loss_mean / iter_precomp)
    return gradient_param, scale_param_task


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    logger = set_up_logger(cfg.OUTPUT_DIR)
    logger.info("Starting gradient evaluation for model {}".format(cfg.MODEL.WEIGHTS))
    start_iter = 0
    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                         use_augmix=cfg.TRAIN.USE_AUGMIX or cfg.TRAIN.REG.CONSISTENCY,
                                                                         width=cfg.TRAIN.AUGMIX_WIDTH))
    with EventStorage(start_iter) as storage:
        iter_precomp = min(100, int(len(data_loader.dataset.dataset) / cfg.SOLVER.IMS_PER_BATCH))
        gradient_param_task, scale_param_task = get_gradscaler(cfg, model, data_loader, iter_precomp, logger)
        with open(os.path.join(cfg.OUTPUT_DIR, "scale_param.json"), "w") as outfile:
            json.dump(scale_param_task, outfile)
        with open(os.path.join(cfg.OUTPUT_DIR, "grad_param.json"), "w") as outfile:
            json.dump(gradient_param_task, outfile)
        with open(os.path.join(cfg.OUTPUT_DIR, "mean_rgn.txt"), "w") as outfile:
            mean_rgn = sum([v for k, v in scale_param_task.items()]) / len(scale_param_task)
            outfile.write('RGN {}'.format(mean_rgn))


def main(args):
    cfg = setup(args, eval=True)
    model = build_model(cfg)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    do_train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
