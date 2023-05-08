# -*- coding: utf-8 -*-
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}


def register_all_pascal_voc(root):
    SPLITS = [
        ("clipart_2012_val", "clipart", "test"),
        ("clipart_2012_test", "clipart", "test"),
        ("clipart_2012_train", "clipart", "train"),
        ("water_2012_train", "VOC_Water", "train"),
        ("water_2012_test", "VOC_Water", "test"),
        ("comic_2012_val", "comic", "test"),
        ("comic_2012_train", "comic", "train"),

    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "foggy_cityscapes_fine_{task}_val": ("foggy_images/val/", "cityscapes/gtFine/val/"),
    "foggy_cityscapes_fine_{task}_test": ("foggy_images/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True,
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


_PREDEFINED_SPLITS_BDD = {
    "bdd_val": ("bdd100k/images/100k/val", "bdd100k/labels/coco_annotation/bdd100k_val_clean_coco.json"),
}


def register_bdd(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata(key.split("_")[0]),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes(_root)
register_all_pascal_voc(_root)
register_bdd(_root)
