import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.blip_captioner import BLIPCaptioner

DEFAULT_CAPTION = "a realistic human hand"


def load_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    """Load metadata from jsonl file"""
    rows: List[Dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_metadata(metadata_path: Path, rows: List[Dict[str, str]]) -> None:
    """Write metadata rows to jsonl file"""
    with metadata_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def needs_caption(row: Dict[str, str], force: bool) -> bool:
    """Check if row needs captioning"""
    if force:
        return True
    caption = row.get("caption", "").strip()
    return caption == "" or caption == DEFAULT_CAPTION


def _collect_caption_targets(
    rows: List[Dict[str, str]],
    dataset_root: Path,
    limit: int,
    force: bool) -> List[Tuple[int, Path]]:
    """Collect rows that need captioning"""
    targets: List[Tuple[int, Path]] = []
    for idx, row in enumerate(rows):
        if limit > 0 and len(targets) >= limit:
            break
        if not needs_caption(row, force):
            continue

        file_name = row.get("file_name", "")
        image_path = dataset_root / file_name
        if image_path.exists():
            targets.append((idx, image_path))
    return targets


def caption_metadata_file(
    dataset_root: Path,
    metadata_path: Path,
    batch_size: int = 8,
    limit: int = 0,
    force: bool = False,
    model_name: str = "Salesforce/blip-image-captioning-base") -> int:
    """Generate captions for images in metadata file"""
    rows = load_metadata(metadata_path)
    targets = _collect_caption_targets(rows, dataset_root, limit, force)

    if not targets:
        print("[caption] No rows need captioning")
        return 0

    captioner = BLIPCaptioner(model_name=model_name)

    print(f"[caption] Captioning {len(targets)} samples")
    updated = 0
    for start in tqdm(range(0, len(targets), batch_size), desc="Captioning"):
        chunk = targets[start : start + batch_size]
        image_paths = [path for _, path in chunk]
        captions = captioner.caption_images(image_paths)

        for (row_idx, _), caption in zip(chunk, captions):
            rows[row_idx]["caption"] = caption
            updated += 1

    write_metadata(metadata_path, rows)
    print(f"[caption] Updated {updated} captions in {metadata_path}")
    return updated


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Caption ControlNet dataset images with BLIP")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/freihand_controlnet"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/freihand_controlnet/metadata.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite all existing captions")
    parser.add_argument("--model-name", type=str, default="Salesforce/blip-image-captioning-base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caption_metadata_file(
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
