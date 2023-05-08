import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from collections.abc import Mapping
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    PascalVOCDetectionEvaluator,
    LVISEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
)
from detectron2.evaluation import DatasetEvaluators, print_csv_format

from src.data import DatasetMapper, build_detection_test_loader
from src.evaluator import PascalVOCDetectionEvaluator


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, tasks=("bbox",), distributed=True))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    logger = logging.getLogger("gdet_training")
    results_file = "./inference/" + '_'.join(cfg.OUTPUT_DIR.split('/')[1:])
    if cfg.TEST.NOISE_ALL:
        corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                       "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                       "brightness", "contrast", "elastic_transform", "pixelate",
                       "jpeg_compression"]
        range_sev = [1, 2, 3, 4, 5]
        for corrupt in corruptions:
            for sev in range_sev:
                data_mapper = DatasetMapper(cfg, is_train=False, eval_aug=corrupt, severity=sev)
                for dataset_name in cfg.DATASETS.TEST:
                    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper)
                    evaluator = get_evaluator(
                        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                    )
                    results_i = inference_on_dataset(model, data_loader, evaluator)
                    results[dataset_name + "{}_{}".format(corrupt, str(sev))] = results_i
                    if comm.is_main_process():
                        logger.info("Evaluation results for {} in csv format:".format(
                            dataset_name + "{}_{}".format(corrupt, str(sev))))
                        print_csv_format(results_i)
                    with open(results_file, "a") as my_file:
                        my_file.write(dataset_name + "{}_{}".format(corrupt, str(sev)))
                        for task, res in results_i.items():
                            if isinstance(res, Mapping):
                                # Don't print "AP-category" metrics since they are usually not tracked.
                                important_res = [(k, v) for k, v in res.items() if "-" not in k]
                                # my_file.write("copypaste: Task: {}".format(task))
                                # my_file.write("copypaste: " + ",".join([k[0] for k in important_res]))
                                my_file.write(
                                    "copypaste: " + ",".join(["{0:.4f} \n".format(k[1]) for k in important_res]))
                            else:
                                my_file.write(f"copypaste: {task}={res}")
                if len(results) == 1:
                    results = list(results.values())[0]
        return results
    else:
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = get_evaluator(
                cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            )
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        if len(results) == 1:
            results = list(results.values())[0]
        return results
