#!/usr/bin/env python3
"""Build the MMSegmentation geospatial tutorial (PDF) from tutorial/chapters/*.md.

    pip install reportlab
    python tools/build_tutorial_pdf.py [--out docs/MMSegmentation_Geospatial_Tutorial.pdf]
"""
import datetime
from pathlib import Path

from _pdfbook import (NUM, ROOT, S, Doc, NextPageTemplate, PageBreak, Paragraph, Spacer, chapter, cm,  # noqa: F401
                      heading, image, md_to_flowables, read, toc_block)

VERSION = '1.0'
DATE = datetime.date(2026, 9, 9).strftime('%d %B %Y')
OUT = ROOT / 'docs' / 'MMSegmentation_Geospatial_Tutorial.pdf'
CH = 'tutorial/chapters/'

story = [NextPageTemplate('cover'),
         Spacer(1, 4.2 * cm),
         Paragraph('Semantic segmentation of geospatial imagery with MMSegmentation', S['cover_t']),
         Spacer(1, 0.6 * cm),
         Paragraph('A course for students and colleagues: environment, data, pipelines, models, training, '
                   'evaluation and GIS-ready prediction', S['cover_s']),
         Paragraph('Running example: <i>Acacia tortilis</i> crown mapping from UAV imagery; further examples for '
                   'buildings, roads and multispectral land cover', S['cover_s']),
         Spacer(1, 1.4 * cm),
         Paragraph(f'Version {VERSION} · {DATE}', S['cover_m']),
         Paragraph('Companion repository: github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping (folder tutorial/)',
                   S['cover_m']),
         Paragraph('Software: MMSegmentation 1.2.2 · MMEngine 0.10.7 · MMCV 2.1.0 · PyTorch 2.6.0 (CUDA 11.8)',
                   S['cover_m']),
         Spacer(1, 3.4 * cm),
         Paragraph('Prepared by Mohamed Barakat A. Gibril<br/>GIS and Remote Sensing Center, Research Institute of '
                   'Sciences and Engineering, University of Sharjah', S['cover_m']),
         Paragraph('Contact: mbgibril@sharjah.ac.ae', S['cover_m']),
         Spacer(1, 1.0 * cm),
         Paragraph('<i>Project datasets referenced in this document are not publicly shareable.</i>', S['cover_m']),
         NextPageTemplate('main'), PageBreak()]
story += toc_block()

# front matter
NUM.chapter('0')
story += [PageBreak()] + heading(1, 'How to use this tutorial')
story += md_to_flowables(read(CH + '00_how_to_use.md').split('\n', 1)[1], skip_h1=True)

chapters = [
    ('1', 'Semantic segmentation and the OpenMMLab stack', '01_foundations.md'),
    ('2', 'The team environment', '02_environment.md'),
    ('3', 'Anatomy of MMSegmentation', '03_anatomy.md'),
    ('4', 'Configuration files', '04_configs.md'),
    ('5', 'From GIS layers to training tiles', '05_data_preparation.md'),
    ('6', 'Datasets, pipelines and augmentation', '06_datasets_pipelines.md'),
    ('7', 'Models: choosing, adapting and switching architectures', '07_models.md'),
    ('8', 'Training', '08_training.md'),
    ('9', 'Evaluation', '09_evaluation.md'),
    ('10', 'Prediction and export to GIS', '10_prediction.md'),
    ('11', 'Worked examples', '11_worked_examples.md'),
    ('12', 'Team workflow', '12_team_workflow.md'),
    ('A', 'Cheat-sheets', 'A_cheatsheets.md'),
    ('B', 'Troubleshooting', 'B_troubleshooting.md'),
    ('C', 'Glossary', 'C_glossary.md'),
    ('D', 'Exercise solutions', 'D_exercises.md'),
]
for label, title, fname in chapters:
    md = read(CH + fname).split('\n', 1)[1]   # drop the file's own H1
    story += chapter(label, title, md)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(OUT))
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Doc(str(out), header='MMSegmentation for geospatial imagery · team tutorial', footer_right=f'v{VERSION} · {DATE}',
        title='Semantic segmentation of geospatial imagery with MMSegmentation', author='Mohamed Barakat A. Gibril',
        subject='Tutorial').multiBuild(story)
    print('written', out, out.stat().st_size // 1024, 'kB')
