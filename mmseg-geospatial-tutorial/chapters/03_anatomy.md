# 3. Anatomy of MMSegmentation

## 3.1 Two copies of the library

When MMSegmentation is installed with `pip install mmsegmentation==1.2.2`, the
Python package `mmseg` is installed together with a hidden folder `mmseg/.mim`
that carries the configuration files and the tools of the repository. A source
checkout (`git clone https://github.com/open-mmlab/mmsegmentation`) has the
same content at the top level. The two are interchangeable; the tutorial image
uses the pip package and vendors `tools/train.py` and `tools/test.py`, so the
source checkout is not needed.

```
mmsegmentation/                (source checkout)      pip package equivalent
├── configs/                   experiment configs      mmseg/.mim/configs/
├── mmseg/                     the library             mmseg/
├── tools/                     command-line scripts    mmseg/.mim/tools/
├── demo/                      inference demos, notebooks
├── docs/                      documentation sources
├── tests/                     unit tests
├── projects/                  community contributions (not part of the package)
├── requirements/              dependency lists
└── model-index.yml            model zoo index used by `mim download`
```

## 3.2 `configs/`: the experiments

```
configs/
├── _base_/
│   ├── datasets/     one file per dataset: pipelines + dataloaders + evaluator (ade20k.py, potsdam.py, loveda.py, ...)
│   ├── models/       one file per architecture family: data_preprocessor + backbone + heads (pspnet_r50-d8.py, upernet_r50.py, segformer_mit-b0.py, ...)
│   ├── schedules/    optimiser + learning-rate schedule + loops + hooks (schedule_20k.py ... schedule_320k.py)
│   └── default_runtime.py   logging, visualiser, checkpoint loading defaults
└── <method>/          full experiments = combination of the four bases + overrides, e.g.
                       segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py
```

The file name of a full experiment encodes model, batch (`8xb2` = 8 GPUs × 2),
iterations, dataset and crop size. The `_base_` folder is the part you reuse:
chapter 4 shows how a project config points to `mmseg::_base_/models/upernet_r50.py`
and changes only what differs.

## 3.3 `mmseg/`: the library

| Sub-package | Content | You touch it when |
|---|---|---|
| `apis/` | `init_model`, `inference_model`, `show_result_pyplot`, `MMSegInferencer`, `RSInferencer` (remote-sensing sliding-window inference) | writing prediction scripts |
| `datasets/` | `BaseSegDataset` and one class per public dataset (`ADE20KDataset`, `PotsdamDataset`, `LoveDADataset`, `ISPRSDataset`, ...); `transforms/` with loading (`LoadImageFromFile`, `LoadAnnotations`, `LoadSingleRSImageFromFile`), augmentation (`RandomCrop`, `RandomFlip`, `PhotoMetricDistortion`, `RandomRotate`, `Albu`, ...) and formatting (`PackSegInputs`) | defining a dataset, changing augmentation |
| `models/backbones/` | encoders: `ResNetV1c`, `HRNet`, `UNet`, `MixVisionTransformer` (SegFormer), `SwinTransformer`, `BEiT`, `MAE`, `MSCAN` (SegNeXt), `TIMMBackbone` (any timm model), ... | choosing/adding an encoder |
| `models/decode_heads/` | decoders: `FCNHead`, `PSPHead`, `ASPPHead`, `DepthwiseSeparableASPPHead` (DeepLabV3+), `UPerHead`, `SegformerHead`, `Mask2FormerHead`, ... | choosing/adding a decoder |
| `models/necks/` | optional feature adapters between backbone and head (`FPN`, `MultiLevelNeck`, `JPU`) | rarely |
| `models/losses/` | `CrossEntropyLoss`, `DiceLoss`, `LovaszLoss`, `FocalLoss`, `TverskyLoss`, `BoundaryLoss` | class imbalance, boundary quality |
| `models/segmentors/` | `EncoderDecoder` (the standard wrapper: backbone → neck → head; `whole` or `slide` inference), `CascadeEncoderDecoder`, `SegTTAModel` | test-time behaviour |
| `models/data_preprocessor.py` | `SegDataPreProcessor`: normalisation, BGR→RGB, padding, batching on the GPU | multispectral inputs, tile sizes |
| `engine/` | hooks (`SegVisualizationHook`), optimiser constructors (layer-wise decay for transformers) | monitoring, fine-tuning schedules |
| `evaluation/metrics/` | `IoUMetric` (IoU, Dice/F-score, precision, recall, accuracy per class), `CityscapesMetric`, `DepthMetric` | reporting |
| `structures/` | `SegDataSample`: the container that carries an image's ground truth, prediction and metadata through the pipeline | writing custom heads or inference code |
| `registry/` | the registries: `MODELS`, `DATASETS`, `TRANSFORMS`, `METRICS`, `HOOKS`, ... | registering custom components |
| `visualization/` | `SegLocalVisualizer`: overlays for logs and demos | |
| `utils/` | class names and palettes of public datasets, misc helpers | |

## 3.4 `tools/`: the command line

| Script | Purpose |
|---|---|
| `tools/train.py <config>` | train; `--work-dir`, `--resume`, `--amp`, `--cfg-options` |
| `tools/test.py <config> <checkpoint>` | evaluate; `--out` (save predictions), `--show-dir` (overlays), `--tta` |
| `tools/analysis_tools/get_flops.py <config> --shape H W` | parameters and FLOPs |
| `tools/analysis_tools/benchmark.py <config> <ckpt>` | inference speed |
| `tools/analysis_tools/browse_dataset.py <config> --output-dir DIR` | render augmented training samples (verify the pipeline) |
| `tools/analysis_tools/confusion_matrix.py <config> <pred dir> <save dir>` | confusion matrix from `test.py --out` |
| `tools/analysis_tools/analyze_logs.py <json log> --keys mIoU` | plot curves from a run's JSON log |
| `tools/misc/print_config.py <config>` | print the fully resolved config |
| `tools/misc/publish_model.py in.pth out.pth` | strip optimiser state, add hash, for release |
| `tools/dataset_converters/*.py` | prepare public datasets (Potsdam, Vaihingen, LoveDA, iSAID, ...) |
| `tools/model_converters/*.py` | convert third-party pretrained weights (Swin, MiT, BEiT, ...) to MMSegmentation keys |

The tutorial adds, in its own `tools/`: `check_dataset.py`, `make_tiles.py`,
`band_statistics.py`, `compare_runs.py`, `adapt_first_conv.py`,
`download_zoo_weights.py`, `geospatial_inference.py`,
`batch_geospatial_inference.py`, `inspect_checkpoint.py`, `verify_install.py`,
plus vendored copies of `train.py` and `test.py` (the latter with
`--test-split` and work-directory checkpoint resolution).

## 3.5 How a training run flows

```
config file  ──►  Runner (MMEngine)
                    ├── build train_dataloader
                    │     dataset (BaseSegDataset: lists img/mask pairs)
                    │       └── pipeline: Load → Augment → PackSegInputs   (CPU workers)
                    │     sampler (InfiniteSampler) + collate  ──► batch
                    ├── model.data_preprocessor: to GPU, BGR→RGB, normalise, pad, stack
                    ├── model(inputs, data_samples, mode='loss')
                    │     backbone → [neck] → decode_head (+ auxiliary_head) → losses
                    ├── optim_wrapper: backward, clip_grad, optimizer.step, [AMP]
                    ├── param_scheduler: learning rate for the next iteration
                    ├── hooks: IterTimer, Logger, Checkpoint, SegVisualization, ...
                    └── every val_interval: ValLoop
                          val_dataloader → model(mode='predict') → IoUMetric → mIoU, mFscore, ...
```

Three consequences that explain most surprises:

1. Augmentation happens in CPU worker processes (`num_workers`); slow training
   with an idle GPU usually means too few workers or a network drive.
2. Normalisation is **not** in the pipeline but in `data_preprocessor`, on the
   GPU; `mean`/`std` must have one entry per input band.
3. Metrics are computed by the evaluator on predictions resized to the original
   image size; test-time `mode='slide'` versus `'whole'` changes accuracy on
   large tiles.

## 3.6 Where your own code plugs in

The tutorial's `geoseg` package registers one additional component in the same
registries, and any project can do the same:

| Registry | Component | Registered name | Source |
|---|---|---|---|
| TRANSFORMS | rasterio multi-band loader | `LoadRasterioImage` | `geoseg/transforms.py` (60 lines) |

A config activates it with `custom_imports = dict(imports=['geoseg'])`. This is
the pattern for any custom component: a Python package, a decorator
(`@TRANSFORMS.register_module()` / `@MODELS.register_module()` /
`@DATASETS.register_module()`), and one line in the config (chapter 7.7 adds
a backbone the same way).

## Exercise 2

Open `mmseg/.mim/configs/_base_/models/deeplabv3plus_r50-d8.py` (path printed
by `python -c "import mmseg, os; print(os.path.dirname(mmseg.__file__))"`).
Identify: the backbone class and its `out_indices`; which feature level the
decode head consumes (`in_index`); which level the auxiliary head consumes; and
the two losses. Then find the same information in the resolved template config with
`python tools/misc/print_config.py templates/project/configs/deeplabv3plus_r50.py`.
