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

import copy
import os

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from eval_utils import do_test
from utils import name_trainmode, set_up_logger, gradient_modifier, get_gradscaler
from src.data import DatasetMapper, build_detection_train_loader


def do_train(cfg, model, resume=False):
    model.train()
    if cfg.TRAIN.SEPRETRAIN:
        for param in model.named_parameters():
            if 'bottom_up' in param[0]:
                if 'se_module' not in param[0]:
                    param[1].requires_grad = False

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                         use_augmix=cfg.TRAIN.USE_AUGMIX,
                                                                         width=cfg.TRAIN.AUGMIX_WIDTH))
    logger = set_up_logger(cfg.OUTPUT_DIR)
    train_mode = name_trainmode(cfg)
    logger.info("Starting training model {} with backbone {}".format(cfg.MODEL.PRETRAIN_NAME, cfg.MODEL.BACKBONE.NAME))
    logger.info("Starting training {} from iteration {}".format(train_mode, start_iter))

    if cfg.TRAIN.REG.RGN or cfg.TRAIN.REG.EWC or cfg.TRAIN.REG.PLAIN_DIST:
        logger.info(f"Starting training {cfg.TRAIN.REG}")
        model_ref = copy.deepcopy(model)
        checkpointer_ref = DetectionCheckpointer(
            model_ref, "tmp", optimizer=optimizer, scheduler=scheduler
        )
        weight_ref = cfg.MODEL.WEIGHTS if cfg.MODEL.REF_WEIGHTS is None else cfg.MODEL.REF_WEIGHTS
        _ = (
                checkpointer_ref.resume_or_load(weight_ref, resume=resume).get("iteration", -1) + 1
        )
        trainer_reg(cfg, model, model_ref, optimizer, scheduler, periodic_checkpointer,
                    start_iter, max_iter, data_loader, writers, logger)
    else:
        trainer_plain(cfg, model, optimizer, scheduler, periodic_checkpointer,
                      start_iter, max_iter, data_loader, writers)


def trainer_reg(cfg, model, model_ref, optimizer, scheduler, periodic_checkpointer,
                start_iter, max_iter, data_loader, writers, logger):
    with EventStorage(start_iter) as storage:
        if start_iter < max_iter:
            iter_precomp = min(cfg.TRAIN.ITER_GRAD_COMP,
                               int(len(data_loader.dataset.dataset) / cfg.SOLVER.IMS_PER_BATCH))
            if not cfg.TRAIN.REG.PLAIN_DIST:
                scale_param_task = get_gradscaler(cfg, model, data_loader, iter_precomp, logger)
            else:
                scale_param_task = None
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            if cfg.TRAIN.USE_AUGMIX:
                for d in data:
                    d['image'] = d['image_aug']
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all()
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            gradient_modifier(cfg, model, model_ref,
                              scale_param_task,
                              optimizer.param_groups[0]["lr"])
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def trainer_plain(cfg, model, optimizer, scheduler, periodic_checkpointer,
                  start_iter, max_iter, data_loader, writers):
    # stop_grad flag will stop gradient for pre-trained model.
    # if we train se module in backbone, we need to backprop gradient into backbone. 
    # This is because se-module is defined in the pre-trained model.
    stop_grad = cfg.TRAIN.PRETRAIN and not cfg.TRAIN.SEPRETRAIN
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data, stop_grad=stop_grad)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all()
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
