"""One-off Klein edits of a single image (rule 4 rewrite by editing), each with a manifest entry.

  python -m pipeline.klein_edit <asset_id> <src.png> <dst.png> <seeds,comma> "<change>" [W H]
Writes <dst> with -<seed> suffixes when several seeds are given. W x H default to the source size rounded to 16.
"""
import sys
from pathlib import Path

from PIL import Image

from pipeline import characters, comfy, manifest, wf
from pipeline.manifest import ROOT


def edit(aid, src, dst, seeds, change, W=None, H=None, stage="5", kind="look_edit"):
    src, dst = Path(src), Path(dst)
    if W is None:
        w, h = Image.open(src).size
        W, H = (w // 16) * 16, (h // 16) * 16
    ref = comfy.upload(src)
    prompt = (f"{characters.STYLE} Recreate Figure 1 exactly: the same subject, design, pose, framing, background and "
              f"lighting. Change only this: {change}")
    outs = []
    for seed in seeds:
        d = dst if len(seeds) == 1 else dst.with_name(f"{dst.stem}-{seed}{dst.suffix}")
        g = wf.klein(f"edit_{d.stem}", prompt, seed=seed, width=W, height=H, refs=[ref], prefix="launch-window/edits/e")
        wfp = g.save(ROOT / "workflows" / "edits")
        res = comfy.run(g.api(), name=aid, timeout=600)
        comfy.download(comfy.output_files(res["outputs"])[0], d)
        a = aid if len(seeds) == 1 else f"{aid}:{seed}"
        manifest.upsert(a, stage=stage, kind=kind, path=str(d.resolve().relative_to(ROOT)), prompt=prompt, seed=seed,
                        model="flux-2-klein-9b-kv-fp8", workflow=str(wfp.relative_to(ROOT)), score=None, status="done",
                        meta={"edit_of": str(src.resolve().relative_to(ROOT)), "change": change})
        outs.append(d)
        print(d, res["seconds"], flush=True)
    return outs


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 5:
        print(__doc__)
        sys.exit(2)
    wh = (int(a[5]), int(a[6])) if len(a) > 6 else (None, None)
    edit(a[0], a[1], a[2], [int(x) for x in a[3].split(",")], a[4], *wh)
    comfy.free()
