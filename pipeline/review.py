"""Review helpers: labelled image grids for subagent review (keeps each reviewer's context small)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parent.parent / "overlays" / "fonts" / "JetBrainsMono-wght.ttf"


def grid(paths, out, labels=None, cols=4, cell_w=480, pad=6) -> str:
    """Tile images into one JPEG with a big label (A, B, C...) in each corner."""
    labels = labels or [chr(65 + i) for i in range(len(paths))]
    ims = [Image.open(p).convert("RGB") for p in paths]
    cells = []
    for im in ims:
        h = int(cell_w * im.height / im.width)
        cells.append(im.resize((cell_w, h), Image.LANCZOS))
    ch = max(c.height for c in cells)
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (ch + pad) + pad), (24, 24, 24))
    try:
        font = ImageFont.truetype(str(FONT), 34)
    except OSError:
        font = ImageFont.load_default()
    for i, (c, lab) in enumerate(zip(cells, labels)):
        x, y = pad + (i % cols) * (cell_w + pad), pad + (i // cols) * (ch + pad)
        sheet.paste(c, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle([x, y, x + 22 + 21 * len(lab), y + 44], fill=(0, 0, 0))
        d.text((x + 8, y + 2), lab, fill=(255, 220, 90), font=font)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return str(out)
