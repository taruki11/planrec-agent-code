from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_dir: Path,
    stage: str,
    model: str,
    semantic_ids: Path,
    data_files: Iterable[Path],
    arguments: dict[str, Any],
    status: str,
) -> None:
    files = [semantic_ids, *data_files]
    parent_manifest = Path(model) / "run_manifest.json"
    parent_manifest_sha256 = file_sha256(parent_manifest) if parent_manifest.is_file() else None
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "total_vram_bytes": properties.total_memory}
    manifest = {
        "pipeline": "planrec_lm_v1",
        "stage": stage,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_input": model,
        "inputs": {str(path): file_sha256(path) for path in files},
        "parent_run_manifest_sha256": parent_manifest_sha256,
        "arguments": arguments,
        "torch_version": torch.__version__,
        "gpu": gpu,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
