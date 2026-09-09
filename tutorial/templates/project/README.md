# Project template

Copy this folder to start a new segmentation project:

```bash
cp -r tutorial/templates/project projects/<my_project>
```

Then edit `configs/_base_/dataset.py` (classes, palette, data root, suffixes)
and train any model config:

```bash
python tools/train.py projects/<my_project>/configs/deeplabv3plus_r50.py --work-dir work_dirs/<my_project>/deeplabv3plus_r50
```

Model configs inherit MMSegmentation's own base configs through the `mmseg::`
prefix, so nothing is copied from the upstream repository. Add or remove model
configs freely; the dataset and schedule are shared.
