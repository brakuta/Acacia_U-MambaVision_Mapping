#!/usr/bin/env python3
"""Build the U-MV technical hand-over report (PDF) from the repository docs.

    pip install reportlab            # fonts: Liberation (serif/sans) and DejaVu Sans Mono
    python tools/build_handover_pdf.py [--out docs/U-MV_Technical_Handover_Guide.pdf]

The report consolidates README.md, docs/01-08, docs/REVISION_NOTES.md and
Pretrained_Weights/README.md into one numbered document with a cover page,
table of contents, PDF outline and a command appendix.
"""
import datetime
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, ListFlowable, ListItem, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'U-MV_Technical_Handover_Guide.pdf'
VERSION = '1.1'
DATE = datetime.date(2026, 9, 9).strftime('%d %B %Y')

# ---------------------------------------------------------------- fonts
FD = '/usr/share/fonts/truetype'
pdfmetrics.registerFont(TTFont('Serif', f'{FD}/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Serif-Bold', f'{FD}/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Serif-Italic', f'{FD}/liberation/LiberationSerif-Italic.ttf'))
pdfmetrics.registerFont(TTFont('Serif-BoldItalic', f'{FD}/liberation/LiberationSerif-BoldItalic.ttf'))
pdfmetrics.registerFont(TTFont('Sans', f'{FD}/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Sans-Bold', f'{FD}/liberation/LiberationSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Mono', f'{FD}/dejavu/DejaVuSansMono.ttf'))
pdfmetrics.registerFont(TTFont('Mono-Bold', f'{FD}/dejavu/DejaVuSansMono-Bold.ttf'))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('Serif', normal='Serif', bold='Serif-Bold', italic='Serif-Italic', boldItalic='Serif-BoldItalic')
registerFontFamily('Mono', normal='Mono', bold='Mono-Bold', italic='Mono', boldItalic='Mono-Bold')

INK = colors.HexColor('#1a1a1a')
ACCENT = colors.HexColor('#2f5d3a')     # acacia green
RULE = colors.HexColor('#9aa79d')
SHADE = colors.HexColor('#eef3ee')
CODEBG = colors.HexColor('#f4f4f2')
GREY = colors.HexColor('#555555')

S = {
    'body': ParagraphStyle('body', fontName='Serif', fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY,
                           spaceAfter=6, textColor=INK),
    'bullet': ParagraphStyle('bullet', fontName='Serif', fontSize=10.5, leading=14, alignment=TA_LEFT,
                             textColor=INK, spaceAfter=2),
    'h1': ParagraphStyle('h1', fontName='Sans-Bold', fontSize=18, leading=22, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=10),
    'h2': ParagraphStyle('h2', fontName='Sans-Bold', fontSize=13, leading=16, textColor=INK,
                         spaceBefore=12, spaceAfter=5),
    'h3': ParagraphStyle('h3', fontName='Sans-Bold', fontSize=11, leading=14, textColor=GREY,
                         spaceBefore=8, spaceAfter=3),
    'code': ParagraphStyle('code', fontName='Mono', fontSize=8.2, leading=10.4, textColor=INK,
                           backColor=CODEBG, borderPadding=(5, 6, 5, 6), leftIndent=0,
                           spaceBefore=4, spaceAfter=8),
    'cell': ParagraphStyle('cell', fontName='Serif', fontSize=8.6, leading=10.6, textColor=INK),
    'cellh': ParagraphStyle('cellh', fontName='Sans-Bold', fontSize=8.6, leading=10.6, textColor=INK),
    'caption': ParagraphStyle('caption', fontName='Serif-Italic', fontSize=9, leading=12,
                              alignment=TA_CENTER, textColor=GREY, spaceBefore=4, spaceAfter=10),
    'note': ParagraphStyle('note', fontName='Serif-Italic', fontSize=9.5, leading=12.5, textColor=GREY,
                           spaceAfter=6),
    'toc1': ParagraphStyle('toc1', fontName='Sans-Bold', fontSize=10.5, leading=15, leftIndent=0),
    'toc2': ParagraphStyle('toc2', fontName='Serif', fontSize=10, leading=13.5, leftIndent=14),
    'cover_t': ParagraphStyle('cover_t', fontName='Sans-Bold', fontSize=24, leading=30, textColor=ACCENT,
                              alignment=TA_LEFT),
    'cover_s': ParagraphStyle('cover_s', fontName='Serif', fontSize=13, leading=18, textColor=INK),
    'cover_m': ParagraphStyle('cover_m', fontName='Serif', fontSize=10.5, leading=15, textColor=GREY),
}
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
AVAIL_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------- inline markdown -> reportlab markup
GLYPH_FIX = [('\u202f', ' '), ('\u00a0', ' ')]
_SUP_CHARS = '\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079'
_SUP = str.maketrans(_SUP_CHARS, '-0123456789')
_SUP_RE = re.compile('[' + _SUP_CHARS + ']*[\u207b\u2070\u2074-\u2079][' + _SUP_CHARS + ']*')


def _superscripts(t: str) -> str:
    # runs of Unicode superscripts containing a minus or a digit outside Latin-1
    return _SUP_RE.sub(lambda m: '<super>' + m.group(0).translate(_SUP) + '</super>', t)


def inline(md: str) -> str:
    t = html.escape(md, quote=False)
    for a, b in GLYPH_FIX:
        t = t.replace(a, b)
    t = _superscripts(t)
    codes = []

    def _stash(m):
        codes.append(f'<font face="Mono" size="8.3">{m.group(1)}</font>')
        return f'\x00{len(codes) - 1}\x00'
    t = re.sub(r'`([^`]+)`', _stash, t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?!\w)', r'<i>\1</i>', t)
    t = re.sub(r'\x00(\d+)\x00', lambda m: codes[int(m.group(1))], t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: f'{m.group(1)}' if m.group(2).startswith(('docs/', 'Pretrained', '#'))
               else f'{m.group(1)} (<font color="#2f5d3a">{m.group(2)}</font>)', t)
    return t


def code_block(lines):
    esc = [html.escape(ln, quote=False).replace(' ', '&nbsp;') or '&nbsp;' for ln in lines]
    longest = max((len(ln) for ln in lines), default=0)
    style = S['code']
    if longest > 88:  # shrink so that wide listings do not wrap mid-word (approx. 0.6 em per char)
        fs = max(6.8, min(8.2, (AVAIL_W - 24) / (longest * 0.615)))
        style = ParagraphStyle('code_small', parent=S['code'], fontSize=fs, leading=fs * 1.27)
    return Paragraph('<br/>'.join(esc), style)


def make_table(rows):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    body = [r + [''] * (ncol - len(r)) for r in body]
    lens = [max(len(r[c]) for r in [header] + body) for c in range(ncol)]
    weights = [min(max(n, 8), 140) ** 0.8 for n in lens]
    tot = sum(weights)
    widths = [AVAIL_W * w / tot for w in weights]
    data = [[Paragraph(inline(c), S['cellh']) for c in header]]
    for r in body:
        data.append([Paragraph(inline(c), S['cell']) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SHADE),
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, ACCENT),
        ('LINEBELOW', (0, -1), (-1, -1), 0.6, RULE),
        ('LINEBELOW', (0, 1), (-1, -2), 0.25, colors.HexColor('#d8ddd8')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    return [t, Spacer(1, 8)]


class Numbering:
    def __init__(self):
        self.ch = 0; self.sec = 0; self.sub = 0; self.label = ''

    def chapter(self, label):
        self.label = label; self.sec = 0; self.sub = 0

    def section(self):
        self.sec += 1; self.sub = 0; return f'{self.label}.{self.sec}'

    def subsection(self):
        self.sub += 1; return f'{self.label}.{self.sec}.{self.sub}'


NUM = Numbering()
STRIP_NUM = re.compile(r'^(\d+(\.\d+)*\.?|[A-Z]\.\d+(\.\d+)*)\s+')


def heading(level, text):
    text = STRIP_NUM.sub('', text.strip())
    if level == 1:
        p = Paragraph(f'{NUM.label}&nbsp;&nbsp;{inline(text)}', S['h1']); p._toc = (0, f'{NUM.label}  {text}')
        return [p]
    if level == 2:
        n = NUM.section()
        p = Paragraph(f'{n}&nbsp;&nbsp;{inline(text)}', S['h2']); p._toc = (1, f'{n}  {text}')
        return [p]
    n = NUM.subsection()
    return [Paragraph(f'{n}&nbsp;&nbsp;{inline(text)}', S['h3'])]


def md_to_flowables(md: str, skip_h1=False):
    out, lines, i = [], md.splitlines(), 0
    para, bullets, num_items, table = [], [], [], []

    def flush_para():
        nonlocal para
        if para:
            out.append(Paragraph(inline(' '.join(para)), S['body'])); para = []

    def flush_lists():
        nonlocal bullets, num_items
        if bullets:
            out.append(ListFlowable([ListItem(Paragraph(inline(b), S['bullet']), leftIndent=12) for b in bullets],
                                    bulletType='bullet', start='•', bulletFontSize=8, leftIndent=14))
            out.append(Spacer(1, 4)); bullets = []
        if num_items:
            out.append(ListFlowable([ListItem(Paragraph(inline(b), S['bullet']), leftIndent=14) for b in num_items],
                                    bulletType='1', bulletFontName='Serif', bulletFontSize=10, leftIndent=16))
            out.append(Spacer(1, 4)); num_items = []

    def flush_table():
        nonlocal table
        if table:
            rows = [[c.strip() for c in r.strip().strip('|').split('|')] for r in table
                    if not re.match(r'^\s*\|?\s*:?-{2,}', r)]
            out.extend(make_table(rows)); table = []

    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            flush_para(); flush_lists(); flush_table()
            j = i + 1; block = []
            while j < len(lines) and not lines[j].startswith('```'):
                block.append(lines[j]); j += 1
            out.append(code_block(block)); i = j + 1; continue
        if ln.startswith('|'):
            flush_para(); flush_lists(); table.append(ln); i += 1; continue
        else:
            flush_table()
        m = re.match(r'^(#{1,3})\s+(.*)', ln)
        if m:
            flush_para(); flush_lists()
            lvl = len(m.group(1))
            if not (lvl == 1 and skip_h1):
                out.extend(heading(lvl, m.group(2)))
            i += 1; continue
        m = re.match(r'^\s*[-*]\s+(.*)', ln)
        if m and not ln.startswith('    '):
            flush_para()
            # continuation lines indented by two spaces
            item = m.group(1); j = i + 1
            while j < len(lines) and re.match(r'^\s{2,}\S', lines[j]) and not re.match(r'^\s*[-*]\s+', lines[j]):
                item += ' ' + lines[j].strip(); j += 1
            bullets.append(item); i = j; continue
        m = re.match(r'^\s*\d+\.\s+(.*)', ln)
        if m:
            flush_para()
            item = m.group(1); j = i + 1
            while j < len(lines) and re.match(r'^\s{2,}\S', lines[j]) and not re.match(r'^\s*\d+\.\s+', lines[j]) and not lines[j].strip().startswith('```'):
                item += ' ' + lines[j].strip(); j += 1
            # code fence inside a numbered item
            if j < len(lines) and lines[j].strip().startswith('```'):
                k = j + 1; block = []
                while k < len(lines) and not lines[k].strip().startswith('```'):
                    block.append(lines[k].strip('\n')[3:] if lines[k].startswith('   ') else lines[k]); k += 1
                num_items.append(item); flush_lists(); out.append(code_block(block)); j = k + 1
            else:
                num_items.append(item)
            i = j; continue
        if ln.startswith('>'):
            flush_para(); flush_lists()
            out.append(Paragraph(inline(ln.lstrip('> ')), S['note'])); i += 1; continue
        if not ln.strip():
            flush_para(); flush_lists(); i += 1; continue
        if ln.startswith('<p') or ln.startswith('</p') or ln.startswith('  <img') or ln.startswith('<img'):
            i += 1; continue
        flush_lists(); para.append(ln.strip()); i += 1
    flush_para(); flush_lists(); flush_table()
    return out


# ---------------------------------------------------------------- document template
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.4 * cm,
                         bottomMargin=2.2 * cm, title='U-MV Acacia tortilis crown mapping: technical hand-over guide',
                         author='Mohamed Barakat A. Gibril', subject='Reproduction and hand-over documentation', **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='f')
        self.addPageTemplates([PageTemplate(id='cover', frames=[frame], onPage=self._cover_page),
                               PageTemplate(id='main', frames=[frame], onPage=self._main_page)])

    def _cover_page(self, canv, doc):
        canv.saveState()
        canv.setFillColor(ACCENT); canv.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 1.2 * cm, stroke=0, fill=1)
        canv.setFillColor(ACCENT); canv.rect(0, 0, PAGE_W, 0.5 * cm, stroke=0, fill=1)
        canv.restoreState()

    def _main_page(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(RULE); canv.setLineWidth(0.5)
        canv.line(MARGIN, PAGE_H - 1.6 * cm, PAGE_W - MARGIN, PAGE_H - 1.6 * cm)
        canv.setFont('Sans', 8); canv.setFillColor(GREY)
        canv.drawString(MARGIN, PAGE_H - 1.45 * cm, 'U-MV · Acacia tortilis crown mapping · Technical hand-over guide')
        canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.45 * cm, f'v{VERSION} · {DATE}')
        canv.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
        canv.drawCentredString(PAGE_W / 2, 1.05 * cm, f'Page {doc.page}')
        canv.restoreState()

    def afterFlowable(self, fl):
        if hasattr(fl, '_toc'):
            level, text = fl._toc
            key = f'h{self.seq.nextf("toc")}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify('TOCEntry', (level, text, self.page, key))


def read(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


def chapter(label, title, md, intro=None, skip_h1=True):
    NUM.chapter(label)
    fl = [PageBreak()] + heading(1, title)
    if intro:
        fl.append(Paragraph(inline(intro), S['body']))
    fl.extend(md_to_flowables(md, skip_h1=skip_h1))
    return fl


def split_md(md, marker):
    a, b = md.split(marker, 1)
    return a, marker + b


# ---------------------------------------------------------------- content
story = []
# cover
story += [Spacer(1, 4.5 * cm),
          Paragraph('U-MV: Regional-scale <i>Acacia tortilis</i> crown mapping from UAV imagery', S['cover_t']),
          Spacer(1, 0.6 * cm),
          Paragraph('Technical hand-over and reproduction guide', S['cover_s']),
          Paragraph('Software, environment, data, training, evaluation and regional inference', S['cover_s']),
          Spacer(1, 1.6 * cm),
          Paragraph(f'Document version {VERSION} · {DATE}', S['cover_m']),
          Paragraph('Repository: github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping', S['cover_m']),
          Paragraph('Reference: Gibril et al. (2026), Int. J. Appl. Earth Obs. Geoinf. 148, 105214, '
                    'doi:10.1016/j.jag.2026.105214', S['cover_m']),
          Spacer(1, 3.5 * cm),
          Paragraph('Prepared by Mohamed Barakat A. Gibril<br/>GIS and Remote Sensing Center, Research Institute of '
                    'Sciences and Engineering, University of Sharjah', S['cover_m']),
          Paragraph('Contact: mbgibril@sharjah.ac.ae', S['cover_m']),
          Spacer(1, 1.2 * cm),
          Paragraph('<i>The dataset described herein is not publicly shareable and is provided for project use only.</i>',
                    S['cover_m'])]
story.insert(0, NextPageTemplate('cover'))
story += [NextPageTemplate('main'), PageBreak()]

# TOC
toc = TableOfContents(); toc.levelStyles = [S['toc1'], S['toc2']]
story += [Paragraph('Contents', S['h1']), toc]

# 1 purpose
NUM.chapter('1')
story += [PageBreak()] + heading(1, 'Purpose and scope of this document')
story += md_to_flowables("""
This document accompanies the hand-over of the U-MV (U-shaped MambaVision) framework for *Acacia tortilis*
crown mapping. It is written for a team member who receives (i) the repository and (ii) a single folder,
`A.tortilis_Data & Model`, containing the training data and the original MMSegmentation work directories, and who
must be able to install the software, verify it, reproduce the published accuracy figures and produce regional
crown maps from new UAV orthomosaics without further assistance.

The document consolidates the repository documentation (`README.md`, `docs/01` to `docs/08`, `docs/REVISION_NOTES.md`)
into one reference. The repository remains the authoritative source; where the two differ, the repository is more
recent.

## How to use this document

- Chapters 2 and 3 describe the framework and the material handed over.
- Chapters 4 to 8 give installation, data, training, evaluation and inference procedures in the order they are needed.
- Chapter 9 is a checklist with an expected outcome for every step; it is the shortest path to a working setup.
- Chapter 10 maps the paper to the configuration files and lists known discrepancies.
- Chapter 11 lists failure modes with remedies. Appendices record what changed in the code revision of September 2026
  and provide a command reference.

## Conventions

Commands are given for a shell inside the Docker container, where the dataset is mounted at `/data` and the checkpoint
folder at `/weights`. Paths on the host are placeholders and must be adapted. Windows paths seen from WSL2 have the form
`/mnt/<drive>/...`; paths containing spaces must be quoted.
""")

# 2 overview
readme = read('README.md')
ov = readme.split('## Highlights', 1)[1].split('## Quick start', 1)[0]
NUM.chapter('2')
story += [PageBreak()] + heading(1, 'Framework overview')
story += md_to_flowables("""
U-MV couples the hierarchical MambaVision encoder of Hatamizadeh and Kautz (2025), a hybrid of Mamba state-space
mixers, self-attention and convolution, with a lightweight U-Net-style decoder. The encoder produces feature maps at
strides 4, 8, 16 and 32; the decoder upsamples the deepest map stage by stage, concatenating skip features and applying
two 3 × 3 convolution–BatchNorm–ReLU blocks per stage, and a 1 × 1 convolution yields two-class logits. Three encoder
sizes are used: tiny (80/160/320/640 channels), small (96/192/384/768) and base (128/256/512/1024). Training minimises
cross-entropy plus three times the Dice loss on 1024 × 1024 UAV tiles at 2.5 cm ground sampling distance.
""")
img = Image(str(ROOT / 'Assets/mamba_vision_architecture.png'))
sc = (AVAIL_W * 0.92) / img.imageWidth
img.drawWidth, img.drawHeight = img.imageWidth * sc, img.imageHeight * sc
story += [img, Paragraph('Figure 1. U-MV architecture: four-stage MambaVision encoder (top) and U-Net decoder with '
                         'skip connections applied to a 1024 × 1024 UAV tile.', S['caption'])]
story += md_to_flowables('## Highlights' + ov)
story += md_to_flowables("""
## Published accuracy

| Model | Validation mIoU | Validation mF-score | Test mIoU | Test mF-score |
|---|---:|---:|---:|---:|
| U-MV-t | 87.91 | 93.20 | 85.44 | 91.61 |
| U-MV-s | 88.02 | 93.27 | 85.38 | 91.58 |
| U-MV-b | 88.11 | 93.32 | 85.30 | 91.52 |

Values in percent (paper, Table 1). On the generalisability set (~11 km², 2 165 tiles) U-MV-s reached 89.48 % mIoU and
94.17 % mF-score. Training on a TITAN RTX (24 GB) with batch size 2 took 9.33 h (tiny), 11.8 h (small) and 18.12 h
(base) for 100 000 iterations.
""")

# 3 what is handed over
h08 = read('docs/08_handover_checklist.md')
part_a, part_b = split_md(h08, '## Step-by-step')
part_a = part_a.split('\n', 1)[1]  # drop title line
story += chapter('3', 'Material handed over', part_a)

# 4-8
story += chapter('4', 'Software environment and installation', read('docs/01_installation.md'))
story += chapter('5', 'Data', read('docs/02_data_preparation.md'))
story += chapter('6', 'Training', read('docs/03_training.md'))
story += chapter('7', 'Evaluation', read('docs/04_evaluation.md'))
story += chapter('8', 'Regional inference on orthomosaics', read('docs/05_inference.md'))
# 9 checklist
story += chapter('9', 'Hand-over procedure (step by step)', part_b)
# 10, 11
story += chapter('10', 'Reproducibility and paper-to-code mapping', read('docs/07_reproducibility.md'))
story += chapter('11', 'Troubleshooting', read('docs/06_troubleshooting.md'))

# appendices
story += chapter('A', 'Revision notes (September 2026)', read('docs/REVISION_NOTES.md'))
story += chapter('B', 'Pretrained weights', read('Pretrained_Weights/README.md'))
story += chapter('C', 'Command reference', """
## Environment

```
cp docker/.env.example docker/.env                 # set DATA_DIR and WEIGHTS_DIR
docker compose --env-file docker/.env -f docker/docker-compose.yml build
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm umv
python tools/verify_install.py --variant small     # stack check + forward pass
python tools/download_backbones.py                 # cache backbones; then export HF_HUB_OFFLINE=1
python -m pytest tests -q                          # unit tests (CPU)
```

## Data and checkpoints

```
python tools/check_dataset.py /data --splits train val test2 Generalizability
python tools/inspect_checkpoint.py "/weights/mambavision-s_generic-unet_acacia-88"
```

## Training

```
python tools/train.py configs/mambavision/U-MV-small.py
python tools/train.py configs/mambavision/U-MV-small.py --resume --work-dir work_dirs/U-MV-small
python tools/train.py configs/mambavision/U-MV-small.py \\
  --cfg-options randomness.seed=0 randomness.deterministic=True
```

## Evaluation

```
CKPT="/weights/mambavision-s_generic-unet_acacia-88"      # folder resolves to best_mIoU_iter_*.pth
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split test2
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split Generalizability
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split test2 \\
  --out preds/ --show-dir vis/
```

## Regional inference

```
CKPT="/weights/mambavision-s_generic-unet_acacia-88"
python tools/geospatial_inference.py \\
  --config configs/mambavision/U-MV-small.py --checkpoint "$CKPT" \\
  --input /data/orthos/block12_cog.tif \\
  --output /data/predictions/block12_crowns.gpkg \\
  --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob

python tools/batch_geospatial_inference.py \\
  --config configs/mambavision/U-MV-small.py --checkpoint "$CKPT" \\
  --input-dir /data/orthos --output-dir /data/predictions \\
  --scratch-dir /tmp/geospatial_work --skip-existing --continue-on-error
```

## Orthomosaic preparation

```
gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512 \\
  -co NUM_THREADS=ALL_CPUS in.tif out_cog.tif
```
""")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(OUT))
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Doc(str(out)).multiBuild(story)
    print('written', out, out.stat().st_size // 1024, 'kB')
