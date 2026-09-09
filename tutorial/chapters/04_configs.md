# 4. Configuration files

## 4.1 A config is a Python file that builds a dictionary

MMEngine executes the config file and keeps every top-level variable. Four
top-level names matter to the `Runner`: `model`, the three dataloaders with
their evaluators, the schedule (`optim_wrapper`, `param_scheduler`,
`train_cfg`, `val_cfg`, `test_cfg`) and the runtime (`default_hooks`,
`env_cfg`, `visualizer`, `log_level`, `load_from`, `resume`, `randomness`).
Everything else (`crop_size`, `norm_cfg`, `metainfo`) is a helper variable.

Every component is a `dict` with a `type` key naming a registered class; the
remaining keys are its constructor arguments:

```python
decode_head = dict(
    type='UPerHead',              # class registered in MODELS
    in_channels=[256, 512, 1024, 2048],
    in_index=[0, 1, 2, 3],
    channels=512,
    num_classes=2,
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
```

`MODELS.build(decode_head)` therefore equals `UPerHead(in_channels=..., ...)`.
The constructor signature is the documentation: `python -c "import inspect;
from mmseg.models.decode_heads import UPerHead; print(inspect.signature(UPerHead.__init__))"`.

## 4.2 Inheritance with `_base_`

A config lists parent files in `_base_`; their dictionaries are merged, and the
child's values override the parents' key by key (deep merge):

```python
_base_ = [
    'mmseg::_base_/models/upernet_r50.py',      # from the installed package
    './_base_/dataset.py',                      # this project's data
    './_base_/schedule.py',                     # this project's schedule
    'mmseg::_base_/default_runtime.py',
]
model = dict(decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
```

Only `num_classes` changes; the backbone, channels and losses come from the
base. The `mmseg::` prefix resolves inside the installed package, so nothing is
copied from upstream and the base is always the version-matched one.

Rules of merging:

| Situation | Behaviour | Remedy |
|---|---|---|
| child key exists in base | value replaced (dicts are merged recursively) | — |
| replace a whole dict instead of merging | add `_delete_=True` inside the child dict | e.g. a different optimiser: `optim_wrapper = dict(_delete_=True, type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01))` |
| lists (`_base_` pipelines, `param_scheduler`, `loss_decode`) | replaced as a whole, never merged | re-state the full list |
| same top-level variable defined in two bases | error "Duplicate key is not allowed among bases" | keep helper variables (e.g. `crop_size`) in one base only |

## 4.3 Reusing base values: `{{_base_.name}}`

A child may reference a variable of a base file with `{{_base_.name}}`. The
value is substituted **after** the child is executed, so it can be used as a
whole value but not in arithmetic:

```python
crop_size = {{_base_.crop_size}}                     # OK: whole value
num_classes = {{_base_.num_classes}}                 # OK
model = dict(data_preprocessor=dict(size=crop_size), test_cfg={{_base_.slide_test_cfg}})
stride = crop_size[0] // 2                           # ERROR: crop_size is still a placeholder string here
```

The project template therefore defines derived values (`num_classes`,
`slide_test_cfg`) in `_base_/dataset.py` and references them whole.

## 4.4 Overriding from the command line

Any key can be overridden without editing files, which is how experiments are
varied and how a config is pointed at another machine's data:

```bash
python tools/train.py projects/buildings/configs/upernet_r50.py \
    --cfg-options train_dataloader.batch_size=2 optim_wrapper.optimizer.lr=5e-5 \
                  train_dataloader.dataset.data_root=/data/buildings_v2 \
                  randomness.seed=1
```

Nested keys use dots; lists use `key="[a,b]"`. The overridden config is what
gets saved in the work directory.

## 4.5 Registries and scopes

Registries are name → class tables, one per component kind (`MODELS`,
`DATASETS`, `TRANSFORMS`, `METRICS`, `HOOKS`, `OPTIM_WRAPPERS`, ...). MMEngine
owns the root registries; MMSegmentation's registries are children with scope
`mmseg`; MMDetection's have scope `mmdet`. When a name exists in several scopes
(e.g. `ResNet` in mmseg, mmdet and mmpretrain) prefix it: `type='mmdet.ResNet'`.
`default_scope = 'mmseg'` in `default_runtime.py` sets the default.

Custom components are registered by importing the module that decorates them:

```python
custom_imports = dict(imports=['umv'], allow_failed_imports=False)
```

`allow_failed_imports=False` turns a typo into an immediate error rather than a
mysterious "X is not in the registry" later.

## 4.6 Inspecting a config

```bash
python tools/misc/print_config.py projects/buildings/configs/upernet_r50.py      # resolved, all bases merged
python tools/misc/print_config.py ... --cfg-options model.decode_head.num_classes=3
```

Every run also writes the resolved config into its work directory
(`work_dirs/<run>/<timestamp>/vis_data/config.py` and next to the log), which
is the file to archive with a result: it records exactly what was run, whatever
the `_base_` files contained at the time.

## 4.7 Reading an unfamiliar config: a checklist

1. `model.type` and `model.backbone.type` — which architecture?
2. `model.backbone.in_channels` and `data_preprocessor.mean/std` — how many
   bands, which normalisation, `bgr_to_rgb`?
3. `decode_head.num_classes` (and `auxiliary_head.num_classes`) — do they
   equal the length of `metainfo['classes']`?
4. `train_pipeline` — crop size, augmentations; `test_pipeline` — resize?
5. `train_cfg.max_iters`, `val_interval`, `optimizer.lr`, `param_scheduler`.
6. `model.test_cfg.mode` — `whole` or `slide` and with which window.
7. `load_from` / `backbone.init_cfg` / `model.pretrained` — where weights come from.
8. `default_hooks.checkpoint.save_best` — which checkpoint is "best".

## Pitfalls

* `SyncBN` (the zoo default) requires distributed training; on one GPU set
  `norm_cfg = dict(type='BN', requires_grad=True)` and pass it to backbone and
  heads, as the templates do.
* `num_classes` is counted **including** background; a binary task has 2.
* `data_preprocessor.size` pads to that size; a crop smaller than the size is
  padded with `pad_val` (image) and `seg_pad_val=255` (mask).
* A `_base_` path is relative to the file that names it, not to the working
  directory.

## Exercise 3

Take `tutorial/templates/project/configs/upernet_r50.py` and, using only
`--cfg-options`, (a) halve the learning rate, (b) change the crop to 768 × 768
(which keys must change together?), (c) disable the auxiliary head. Verify each
with `print_config.py`.
