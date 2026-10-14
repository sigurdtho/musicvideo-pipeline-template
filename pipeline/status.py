"""One-screen pipeline digest (cheap check-ins): python -m pipeline.status"""
import json
import subprocess
from collections import Counter

from pipeline import manifest
from pipeline.manifest import ROOT


def main():
    m = manifest.load()
    shots = json.load(open(ROOT / "video" / "shotlist.json"))["shots"]
    need = {s["id"] for s in shots if s["kind"] in ("i2v", "lipsync") and not s.get("reuse_still")}
    stills = [a for a in m["assets"].values() if a.get("kind") == "still"]
    picked = {a["meta"]["shot"]: a for a in stills if a["status"] == "picked"}
    prov = sum(1 for a in picked.values() if a["meta"].get("provisional"))
    clips = Counter(a["status"] for a in m["assets"].values() if a.get("kind") == "clip")
    i2v = {s["id"] for s in shots if s["kind"] == "i2v"}
    clip_shots = {a["meta"]["shot"] for a in m["assets"].values() if a.get("kind") == "clip" and a["status"] in ("done", "picked")}
    lips = Counter(a["status"] for a in m["assets"].values() if a.get("kind") == "lipsync")
    sync = [round(a["score"]["syncnet_confidence"], 2) for a in m["assets"].values() if a.get("kind") == "lipsync" and a.get("score")
            and a["score"].get("syncnet_confidence") is not None]
    ov = sum(1 for a in m["assets"].values() if a.get("kind") == "overlay")
    mix = m["assets"].get("mix:master", {})
    units = subprocess.run(["systemctl", "--user", "list-units", "__SLUG__-*", "--no-legend", "--state=active"], capture_output=True,
                           text=True).stdout.split()
    active = [u.replace("__SLUG__-", "").replace(".service", "") for u in units if u.startswith("__SLUG__-")]
    owner = (ROOT / ".gpu.owner").read_text().strip() if (ROOT / ".gpu.owner").exists() else "-"
    print(f"stills picked {len(picked & need if isinstance(picked, set) else set(picked) & need)}/{len(need)} (provisional {prov}) | "
          f"clips {len(clip_shots & i2v)}/{len(i2v)} {dict(clips)} | lipsync {dict(lips)} sync {sync} | overlays {ov} | "
          f"mix {mix.get('status')} {mix.get('score')}")
    print(f"active: {active} | gpu: {owner}")
    errs = subprocess.run(f"grep -l -E 'Traceback|cycle error' {ROOT}/qc/logs/*.log 2>/dev/null | xargs -r ls -t | head -3",
                          shell=True, capture_output=True, text=True).stdout.split()
    print("logs with errors (newest first):", [e.split("/")[-1] for e in errs])


if __name__ == "__main__":
    main()
