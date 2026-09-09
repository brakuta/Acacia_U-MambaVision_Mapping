"""Minimal Markdown-to-PDF book builder (ReportLab) shared by the documentation scripts.

Supports: # / ## / ### headings (auto-numbered per chapter), paragraphs, bullet
and numbered lists (numbering continues across code blocks), pipe tables,
fenced code blocks, inline code/bold/italic/links, block quotes, images
(via ``image()``), a cover page, table of contents and PDF outline.
"""
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
from reportlab.platypus import (BaseDocTemplate, Frame, Image, ListFlowable, ListItem, NextPageTemplate,  # noqa: F401
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]

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
    num_next = [1]  # running start for numbered lists interrupted by code blocks

    def flush_para():
        nonlocal para
        if para:
            out.append(Paragraph(inline(' '.join(para)), S['body'])); para = []; num_next[0] = 1

    def flush_lists():
        nonlocal bullets, num_items
        if bullets:
            out.append(ListFlowable([ListItem(Paragraph(inline(b), S['bullet']), leftIndent=12) for b in bullets],
                                    bulletType='bullet', start='•', bulletFontSize=8, leftIndent=14))
            out.append(Spacer(1, 4)); bullets = []
        if num_items:
            out.append(ListFlowable([ListItem(Paragraph(inline(b), S['bullet']), leftIndent=14) for b in num_items],
                                    bulletType='1', bulletFontName='Serif', bulletFontSize=10, leftIndent=16,
                                    start=num_next[0]))
            out.append(Spacer(1, 4)); num_next[0] += len(num_items); num_items = []

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
            num_next[0] = 1
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
    def __init__(self, path, header='', footer_right='', title='', author='', subject='', **kw):
        self.header_text, self.footer_right = header, footer_right
        super().__init__(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.4 * cm,
                         bottomMargin=2.2 * cm, title=title, author=author, subject=subject, **kw)
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
        canv.drawString(MARGIN, PAGE_H - 1.45 * cm, self.header_text)
        canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.45 * cm, self.footer_right)
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


def image(path, width_frac=0.92):
    img = Image(str(path))
    sc = (AVAIL_W * width_frac) / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * sc, img.imageHeight * sc
    return img


def toc_block():
    toc = TableOfContents()
    toc.levelStyles = [S['toc1'], S['toc2']]
    return [Paragraph('Contents', S['h1']), toc]


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
