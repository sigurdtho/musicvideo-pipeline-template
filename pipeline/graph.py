"""Tiny builder for ComfyUI API-format graphs: g.node(cls, **inputs) returns a ref; ref[i] is output i."""
import json
from pathlib import Path


class Ref:
    def __init__(self, nid: str):
        self.id = nid

    def __getitem__(self, i: int):
        return [self.id, i]


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, dict] = {}

    def node(self, class_type: str, _title: str | None = None, **inputs) -> Ref:
        nid = str(len(self.nodes) + 1)
        # Dynamic-combo sub-inputs use dotted keys ("format.quality"); pass them as format__quality.
        fixed = {k.replace("__", "."): v for k, v in inputs.items() if v is not None}
        self.nodes[nid] = {"class_type": class_type, "inputs": fixed}
        if _title:
            self.nodes[nid]["_meta"] = {"title": _title}
        return Ref(nid)

    def api(self) -> dict:
        return self.nodes

    def save(self, folder: str | Path) -> Path:
        """Record the exact API graph used (workflows/<name>.json) for the manifest's `workflow` field."""
        p = Path(folder) / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.nodes, indent=1, ensure_ascii=False))
        return p
