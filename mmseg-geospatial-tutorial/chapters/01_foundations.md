# 1. Semantic segmentation and the OpenMMLab stack

## 1.1 What semantic segmentation does

*Semantic segmentation* assigns a class label to every pixel of an image. For
mapping it is the operation that turns an orthomosaic into a thematic raster:
tree crown / not crown, building / road / water / bare soil, and so on. The
output has the same size as the input and, when the input is georeferenced, the
prediction inherits its coordinate reference system and can be vectorised
directly into GIS layers.

Contrast with two neighbouring tasks. *Object detection* returns bounding boxes
(useful for counting trees, poor for area). *Instance segmentation* returns a
separate mask per object (Mask R-CNN, Mask2Former in instance mode); it is
needed when touching objects must be separated, at a higher annotation and
computational cost. Semantic segmentation is the right choice when the map is
the product: area, extent, cover fraction, or a raster layer for further GIS
analysis.

## 1.2 How a segmentation network is organised

Almost every modern architecture is an *encoder–decoder*:

* The **encoder** (backbone) reduces spatial resolution stage by stage (strides
  4, 8, 16, 32) while increasing the number of feature channels. It is usually
  pretrained on ImageNet, which is why 5 000 tiles can be enough to train a
  good model. Encoders differ in their inductive bias: convolutional networks
  (ResNet, ConvNeXt, U-Net) favour local texture; transformers (Swin, MiT,
  ViT) model long-range context; state-space models (VMamba, MambaVision)
  offer long-range modelling at lower cost.
* The **decoder** (decode head) fuses the multi-scale features back to a
  pixel-wise prediction: ASPP (DeepLabV3+), pyramid pooling (PSPNet), feature
  pyramid (UPerNet), lightweight MLP (SegFormer), U-Net skip connections.
* The **loss** compares the prediction with the reference mask: cross-entropy
  (per-pixel classification), Dice (region overlap, robust to imbalance),
  Lovasz (a surrogate of IoU), Focal (down-weights easy pixels).

Figure 1 shows the structure. A published example on the running dataset
(Gibril et al., 2026) combined one encoder with four different decoders and
found that the encoder determined accuracy far more than decoder complexity,
which is a useful prior when choosing where to spend compute.

![Figure 1. The encoder–decoder structure shared by the architectures in this tutorial: the encoder produces features at strides 4, 8, 16 and 32; the decoder fuses them (here with skip connections) into per-pixel class scores at the input resolution.](docs/figures/encoder_decoder.png)

## 1.3 Why a framework, and why MMSegmentation

Writing training loops by hand is instructive once and error-prone thereafter.
A framework fixes the pieces that are the same for every project: data loading
with augmentation, distributed and mixed-precision training, checkpointing,
logging, evaluation, and a model zoo with pretrained weights. MMSegmentation is
the segmentation library of the OpenMMLab ecosystem:

| Package | Role | Version used |
|---|---|---|
| **MMEngine** | training engine: `Runner`, hooks, config system, registries, checkpoints, logging | 0.10.7 |
| **MMCV** | computer-vision operators (compiled CUDA ops), image transforms, `ConvModule` etc. | 2.1.0 |
| **MMSegmentation** | datasets, pipelines, backbones, decode heads, losses, metrics for semantic segmentation | 1.2.2 |
| MMDetection | detection and instance segmentation; provides the Mask2Former pixel decoder used by MMSegmentation's Mask2Former | 3.3.0 (optional) |
| MMPretrain | classification backbones (ConvNeXt, ...) importable into MMSegmentation | 1.2.0 (optional) |

Its strengths for geospatial work: 40+ architectures with one training script;
declarative configuration files that document an experiment completely;
sliding-window inference for large images; metrics per class; and an easy
route to add custom components (a new backbone or loader is registered with
one decorator and one config line). Its weaknesses: the 8-bit RGB assumption in several transforms
(addressed in chapter 6), and a learning curve caused by the config system
(chapter 4 exists to flatten it).

## 1.4 Vocabulary

| Term | Meaning |
|---|---|
| tile / patch | a fixed-size crop (e.g. 512 × 512 or 1024 × 1024 px) of an orthomosaic used as one training sample |
| mask / annotation / ground truth | single-band raster with the class index of every pixel |
| ignore index | label value (255) excluded from loss and metrics: unlabeled or padded pixels |
| iteration | one optimiser step on one batch; MMSegmentation schedules are iteration-based |
| epoch | one pass over the training set (= tiles / batch size iterations) |
| mIoU | mean over classes of intersection-over-union; the standard segmentation metric |
| checkpoint | saved model weights (and optimiser state) at a given iteration |
| config | the Python file that defines data, model, schedule and runtime of one experiment |
| registry | a name → class table; `type='PSPHead'` in a config is looked up in the model registry |
| work directory | the folder where a run writes logs, checkpoints and the resolved config |

## Exercise 1

The running example dataset was trained on 1024 × 1024 tiles at 2.5 cm GSD
with batch size 2 for 100 000 iterations. How many epochs is that for 26 615
tiles? How many for a 4 893-tile subset? What does this imply for the
iteration budget when you reuse the schedule on the smaller set? (Answer in
Appendix D.)
