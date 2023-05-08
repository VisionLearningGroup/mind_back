# Usage for training

MODEL.PRETRAIN_NAME is config for backbone.
We mainly used three models in a paper, fbnet: ResNet50 instagram model, convnext_base_21k: ConvNeXt model, and
tf_efficientnet_b2_ns: EfficientNet model.

## Plain training

```
python tools/train.py --config configs/Pascal/base.yaml --num-gpus 1 \\
OUTPUT_DIR tmp MODEL.PRETRAIN_NAME fbnet
```

## Training with SE-Block

### Pre-training decoder with SE-Block

Run following commands in this directory.
The flag TRAIN.SEPRETRAIN is effective only for fbnet and convnext_base_21k models.

```
python tools/train.py --config configs/Pascal/base.yaml --num-gpus 1 \\
OUTPUT_DIR tmp MODEL.PRETRAIN_NAME fbnet TRAIN.SEPRETRAIN True
```

### Fine-tuning

Add "_senet" to the MODEL.PRETRAIN_NAME, e.g., fbnet => fbnet_senet.

```
## Example: Train with RGN regularization
python tools/train.py --config configs/Pascal/base.yaml --num-gpus 1 \\
OUTPUT_DIR path_to_dir MODEL.PRETRAIN_NAME fbnet_senet MODEL.WEIGHTS path_model TRAIN.REG.RGN True \\
SOLVER.COEFF 0.1
```

## Training with NAS-FPN

Use configs/Pascal/nas_fb_resnet.yaml. 
