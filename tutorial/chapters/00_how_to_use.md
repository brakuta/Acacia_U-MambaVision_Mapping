# How to use this tutorial

## Who it is for

Three readers, three paths.

| Reader | Goal | Path |
|---|---|---|
| A student starting a segmentation project | train a first model on own tiles within a week | Chapters 2, 5, 6, 8, 9 in order, then the exercises of Appendix D |
| A research assistant taking over a project | reproduce, retrain, compare and deploy models | Chapters 3, 4, 7, 8, 9, 10, 12 |
| A colleague evaluating MMSegmentation for a new task | understand what changes between tasks and sensors | Chapters 1, 4, 6, 7, 11 |

Prerequisites: Python at the level of writing functions and reading tracebacks;
familiarity with raster GIS concepts (CRS, pixel size, bands, NoData); a basic
idea of what a convolutional network is. Nothing about MMSegmentation is assumed.

## Learning outcomes

After working through the tutorial the reader can:

1. explain what each folder of MMSegmentation contains and how a training run
   flows through the library;
2. prepare image–mask tiles from GIS layers and verify them;
3. write a dataset configuration for 8-bit RGB and for 16-bit multispectral data,
   including augmentation pipelines;
4. select, adapt and train at least three different architectures on the same
   data and compare them with consistent metrics;
5. diagnose common failures (data, memory, convergence) from logs;
6. produce georeferenced predictions and vector outputs for GIS.

## Conventions

* Commands are shown for the team's container (chapter 2): the repository is at
  `/workspace/U-MV`, data at `/data`, checkpoints at `/weights`. On a bare
  Linux/conda installation replace these with local paths.
* Configuration snippets are Python (MMSegmentation configs are Python files).
  `...` marks omitted lines.
* **Checkpoint** boxes state what should be observed before continuing.
  **Pitfall** boxes record verified failure modes. **Exercise** boxes propose
  short tasks; solutions are in Appendix D.
* Terms in *italics* at first use are defined in the glossary (Appendix C).

## Versions

MMSegmentation 1.2.2, MMEngine 0.10.7, MMCV 2.1.0, PyTorch 2.6.0 (CUDA 11.8).
Every signature and file path in this tutorial was checked against these
versions; the OpenMMLab 0.x API (``mmseg 0.30`` and earlier) is different and
its tutorials do not apply.
