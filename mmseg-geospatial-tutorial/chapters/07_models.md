# 7. Models: choosing, adapting and switching architectures

## 7.1 A map of the zoo

| Family | Backbone → head | Config base (`mmseg::_base_/models/`) | Pretrained encoder | Character |
|---|---|---|---|---|
| U-Net | `UNet` → `FCNHead` | `fcn_unet_s5-d16.py` | none (from scratch) | small, sharp boundaries, needs more iterations; good for few classes and thin objects |
| PSPNet | `ResNetV1c` → `PSPHead` | `pspnet_r50-d8.py` | ImageNet (`open-mmlab://resnet50_v1c`) | pyramid pooling for context |
| DeepLabV3+ | `ResNetV1c` → `DepthwiseSeparableASPPHead` | `deeplabv3plus_r50-d8.py` | ImageNet | atrous context + low-level skip; the classic strong CNN baseline |
| UPerNet | `ResNetV1c` / `SwinTransformer` / `ConvNeXt` → `UPerHead` | `upernet_r50.py`, `upernet_swin.py`, `upernet_convnext.py` | ImageNet | feature-pyramid fusion; the standard head for transformer encoders |
| SegFormer | `MixVisionTransformer` → `SegformerHead` | `segformer_mit-b0.py` (+ B1–B5 overrides) | ImageNet (MiT) | light transformer, efficient at 1024²; competitive in the Acacia study |
| SegNeXt | `MSCAN` → `LightHamHead` | `segnext_mscan-t` configs | ImageNet | convolutional attention, efficient |
| Mask2Former | any → `Mask2FormerHead` | `mask2former` configs (needs MMDetection) | ImageNet | mask-classification decoder; strong but heavy, slow to train |
| Segmenter | `VisionTransformer` → `SegmenterMaskTransformerHead` | `segmenter_vit-b16_mask.py` | ImageNet | pure ViT |
| any timm model | `TIMMBackbone` → any head | (write the config; see 7.7) | timm (ImageNet) | ConvNeXt, EfficientNet, RegNet, ... without conversion |

Guidance from a published benchmark on the running dataset (Gibril et al.,
2026; test mIoU): SegFormer-B2 85.2, Mask2Former-R50 84.9, U-Net-R50 84.2,
PSPNet-R50 81.9, DeepLabV3+-R50 81.1; a hybrid Mamba–transformer encoder with
a light U-Net decoder reached 85.3–85.4. Transformers and hybrids generalised
better to an unseen region. Start a new project with two baselines from
different families (e.g. DeepLabV3+ and SegFormer-B2) and let the comparison
decide.

## 7.2 Adapting a zoo model to your data

Every adaptation is the same four edits, shown for UPerNet-R50:

```python
_base_ = ['mmseg::_base_/models/upernet_r50.py', './_base_/dataset.py', './_base_/schedule.py',
          'mmseg::_base_/default_runtime.py']
crop_size = {{_base_.crop_size}}
num_classes = {{_base_.num_classes}}
norm_cfg = dict(type='BN', requires_grad=True)            # 1. single-GPU normalisation
model = dict(
    data_preprocessor=dict(size=crop_size),               # 2. tile size (mean/std/bgr_to_rgb for non-RGB)
    backbone=dict(norm_cfg=norm_cfg),                     #    (+ in_channels for multispectral)
    decode_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),      # 3. classes
    auxiliary_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    test_cfg={{_base_.slide_test_cfg}})                   # 4. inference mode
```

The auxiliary head is a second, shallow classifier on an intermediate feature
level used only during training (loss weight 0.4); it stabilises deep CNNs and
is absent from SegFormer. Set `auxiliary_head=None` to drop it.

## 7.3 Pretrained weights

| Mechanism | Where | Applies to |
|---|---|---|
| `model.pretrained='open-mmlab://resnet50_v1c'` or a URL/path | zoo CNN bases | whole backbone, loaded at `init_weights()` |
| `backbone.init_cfg=dict(type='Pretrained', checkpoint=URL_or_path)` | transformer bases (MiT, Swin, ConvNeXt) | backbone |
| `load_from='work_dirs/.../best_mIoU_iter_x.pth'` | top level | whole segmentor; fine-tune a previous run on new data (classes must match, or set `strict=False` semantics by deleting the head keys) |
| `resume=True` / `--resume` | top level | continue the same run (weights + optimiser + iteration) |
| `TIMMBackbone(pretrained=True)` | timm encoders | weights downloaded by timm (Hugging Face Hub) |

Download once for offline hosts: `python tools/download_zoo_weights.py`
(ResNet-50/101 V1c, MiT-B1/B2, Swin-T) and point the config to the local file.
Encoder weights for SegFormer/Swin come converted to MMSegmentation key names
from `download.openmmlab.com`; weights from timm or torchvision require the
converters in `tools/model_converters/`.

## 7.4 Losses for imbalance and boundaries

```python
loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
             dict(type='DiceLoss', loss_weight=3.0)]              # CE + Dice, weighted towards Dice
```

| Loss | When | Notes |
|---|---|---|
| `CrossEntropyLoss` | always, as the base term | `avg_non_ignore=True` averages over labeled pixels only |
| `DiceLoss` | small or sparse targets (crowns, buildings) | region overlap; weight 1–3 relative to CE |
| `LovaszLoss(loss_type='multi_class', reduction='none')` | thin, connected structures (roads, rivers) | direct IoU surrogate; use after a few thousand CE iterations or together with CE |
| `FocalLoss(use_sigmoid=True)` | many easy background pixels | binary/one-vs-rest formulation |
| `TverskyLoss` | trade recall against precision | `alpha`, `beta` |
| `BoundaryLoss` | boundary accuracy (with `GenerateEdge`) | add with small weight |

Each loss needs a distinct `loss_name` only when the same type appears twice.

**Pitfall (verified).** `CrossEntropyLoss(class_weight=[...])` indexes the
weight vector by every label value when averaging, so any 255 pixel — present
whenever padding or unlabeled areas exist — raises an index error in 1.2.2.
Handle imbalance with Dice/Lovasz/Focal or tiling-time down-sampling instead.

## 7.5 Switching between models

Because dataset and schedule are shared bases, switching architecture is
switching one file:

```bash
for m in unet_fcn deeplabv3plus_r50 upernet_r50 segformer_b2; do
  python tools/train.py projects/buildings/configs/$m.py --work-dir work_dirs/buildings/$m
done
python tools/compare_runs.py work_dirs/buildings          # one table, best/last/test metrics
```

`templates/project/scripts/run_experiments.sh` does exactly this and
appends the test-split evaluation. Compare at equal iterations, equal crop and
equal augmentation; report parameters and FLOPs
(`tools/analysis_tools/get_flops.py <config> --shape 1024 1024`) next to
accuracy, as in the paper's efficiency figure.

## 7.6 Size and speed

Measured with `mmengine.analysis.get_model_complexity_info` on the template
configs (2 classes; FLOPs at 512² and scaled to 1024²):

| Model | Params | FLOPs @512² | FLOPs @1024² | Notes |
|---|---|---|---|---|
| SegFormer-B1 | 13.7 M | 15 G | 0.06 T | fastest transformer |
| SegFormer-B2 | 24.7 M | 25 G | 0.10 T | best accuracy/cost among the zoo transformers on the example data |
| U-Net (s5-d16) | 29.1 M | 203 G | 0.81 T | full-resolution decoder is costly |
| DeepLabV3+-R50 (d8) | 43.6 M | 176 G | 0.71 T | output stride 8 is expensive at 1024² |
| UPerNet-R50 | 66.4 M | 237 G | 0.95 T | |

FLOPs scale linearly with pixels for CNNs and roughly so for the windowed and
reduced-attention transformers used here; a 1024² tile costs about 4× a 512²
tile.

## 7.7 Adding an encoder without writing one

`TIMMBackbone` wraps any timm model that exposes feature maps, which covers
most modern encoders (ConvNeXt, EfficientNet, RegNet, ResNeSt, ...):

```python
model = dict(
    backbone=dict(_delete_=True, type='TIMMBackbone', model_name='convnext_tiny', features_only=True,
                  pretrained=True, out_indices=(0, 1, 2, 3)),          # strides 4/8/16/32
    decode_head=dict(in_channels=[96, 192, 384, 768]),                 # ConvNeXt-T widths
    auxiliary_head=dict(in_channels=384))
```

Channel widths come from `timm.create_model(name, features_only=True).feature_info.channels()`.

## 7.8 Adding a custom architecture

A custom backbone is a class that (1) subclasses `mmengine.model.BaseModule`,
(2) implements `forward(x)` returning a list of feature maps at strides
4/8/16/32, (3) carries `@MODELS.register_module()`, and (4) is exposed with
`custom_imports`. The head then only needs the channel widths
(`in_channels=[...]`). A custom head subclasses `BaseDecodeHead`, implements
`forward(inputs)` returning logits at stride 4 and inherits loss computation,
resizing and prediction. Published research code built this way on the
running dataset (a MambaVision encoder wrapped in ~100 lines and a U-Net head
in ~90) is available separately from its authors.

## Exercise 6

Create `projects/<yours>/configs/segformer_b1.py` from `segformer_b2.py`
(MiT-B1: `embed_dims=64`, `num_layers=[2, 2, 2, 2]`, checkpoint `mit_b1`).
Compare parameters and FLOPs of B1, B2 and DeepLabV3+-R50 with
`get_flops.py` at 512 × 512.
