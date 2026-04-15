import argparse
import json
import math
import os
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.caption_dataset import caption_metadata_file
from src.utils.video_to_skeleton import HandSkeletonizer

DEFAULT_OUTPUT_SIZE = 512
DEFAULT_CAPTION = "a hand gesture"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_thread_local = threading.local()

def ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def clear_numbered_pngs(directory: Path) -> int:
    """Delete numbered PNG files so reruns stay deterministic."""
    deleted = 0
    if not directory.exists():
        return deleted
    for path in directory.glob("*.png"):
        if path.stem.isdigit():
            path.unlink()
            deleted += 1
    return deleted


def extract_zip(archive_path: Path, extract_to: Path) -> None:
    """Extract zip archive into target directory."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if extract_to.exists():
        shutil.rmtree(extract_to)

    ensure_dir(extract_to)
    print(f"[extract] Extracting {archive_path} -> {extract_to}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_to)
    print("[extract] Completed")


def _ensure_extracted(archive_path: Path, extracted_root: Path) -> None:
    """Extract the archive if the destination directory is empty."""
    if extracted_root.exists() and any(extracted_root.iterdir()):
        return
    extract_zip(archive_path, extracted_root)


def collect_labeled_images(dataset_root: Path) -> List[Tuple[str, Path]]:
    """Collect images in deterministic class/name order."""
    rows: List[Tuple[str, Path]] = []
    class_dirs: List[Path] = []
    with os.scandir(dataset_root) as entries:
        for entry in entries:
            if entry.is_dir():
                class_dirs.append(Path(entry.path))

    class_dirs.sort(key=lambda path: path.name.lower())

    for class_dir in class_dirs:
        image_paths: List[Path] = []
        with os.scandir(class_dir) as entries:
            for entry in entries:
                if entry.is_file() and Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(Path(entry.path))

        image_paths.sort(key=lambda path: path.name.lower())
        for image_path in image_paths:
            rows.append((class_dir.name, image_path))

    if not rows:
        raise RuntimeError(f"No class-folder images found under: {dataset_root}")

    return rows


def _get_thread_skeletonizer(
    hand_landmarker_model_path: Path,
    output_size: int,
    line_thickness: int,
    joint_radius: int) -> HandSkeletonizer:

    skeletonizer = getattr(_thread_local, "skeletonizer", None)
    if skeletonizer is None:
        skeletonizer = HandSkeletonizer(
            hand_landmarker_model_path=hand_landmarker_model_path,
            output_size=output_size,
            line_thickness=line_thickness,
            joint_radius=joint_radius,
        )
        skeletonizer.open()
        _thread_local.skeletonizer = skeletonizer
    return skeletonizer


def _render_and_write(
    image_path_str: str,
    rgb_out_path_str: str,
    cond_out_path_str: str,
    output_size: int,
    hand_landmarker_model_path: Path,
    line_thickness: int,
    joint_radius: int) -> None:

    rgb_bgr = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Unable to read image: {image_path_str}")

    rgb_resized = cv2.resize(rgb_bgr, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    skeletonizer = _get_thread_skeletonizer(hand_landmarker_model_path, output_size, line_thickness, joint_radius)
    skeleton_bgr = skeletonizer.render_skeleton_from_bgr(rgb_resized)

    if not cv2.imwrite(rgb_out_path_str, rgb_resized):
        raise RuntimeError(f"Failed to write image: {rgb_out_path_str}")
    if not cv2.imwrite(cond_out_path_str, skeleton_bgr):
        raise RuntimeError(f"Failed to write conditioning image: {cond_out_path_str}")


def _process_batch(args: Tuple[int, Sequence[Tuple[int, str, str, str, int]], str, int, int, int, Path]) -> Tuple[int, int]:
    batch_idx, batch_tasks, shard_metadata_path, output_size, line_thickness, joint_radius, hand_landmarker_model_path = args

    processed = 0
    shard_metadata_file = Path(shard_metadata_path)
    with shard_metadata_file.open("w", encoding="utf-8") as meta_fp:
        for _, image_path_str, rgb_out_path_str, cond_out_path_str, _ in batch_tasks:
            try:
                _render_and_write(
                    image_path_str=image_path_str,
                    rgb_out_path_str=rgb_out_path_str,
                    cond_out_path_str=cond_out_path_str,
                    output_size=output_size,
                    hand_landmarker_model_path=hand_landmarker_model_path,
                    line_thickness=line_thickness,
                    joint_radius=joint_radius,
                )
            except Exception as exc:
                print(f"[preprocess] Skipping {image_path_str}: {exc}")
                continue

            out_name = Path(rgb_out_path_str).name
            row = {
                "file_name": str(Path("images") / out_name).replace("\\", "/"),
                "conditioning_image": str(Path("conditioning_images") / out_name).replace("\\", "/"),
                "caption": DEFAULT_CAPTION,
            }
            meta_fp.write(json.dumps(row) + "\n")
            processed += 1

    return batch_idx, processed


def preprocess_hagrid(
    archive_path: Path,
    extracted_root: Path,
    output_root: Path,
    output_size: int,
    line_thickness: int,
    joint_radius: int,
    limit: int,
    num_workers: int = 0,
    auto_caption: bool = True,
    caption_batch_size: int = 8,
    caption_limit: int = 0,
    caption_force: bool = False,
    caption_model_name: str = "Salesforce/blip-image-captioning-base",
    hand_landmarker_model_path: Path = Path("models/hand_landmarker.task")) -> None:
    """Preprocess HAGRID archive for ControlNet training."""

    print("[preprocess] Preparing input archive")
    _ensure_extracted(archive_path, extracted_root)

    top_level_dirs = [path for path in extracted_root.iterdir() if path.is_dir()]
    if not top_level_dirs:
        raise RuntimeError(f"No class folders found in extracted archive: {extracted_root}")

    if len(top_level_dirs) == 1:
        dataset_root = top_level_dirs[0]
    else:
        dataset_root = extracted_root

    labeled_images = collect_labeled_images(dataset_root)
    if limit > 0:
        labeled_images = labeled_images[:limit]

    images_out = output_root / "images"
    cond_out = output_root / "conditioning_images"
    ensure_dir(images_out)
    ensure_dir(cond_out)

    metadata_path = output_root / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    shard_dir = output_root / "metadata_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    ensure_dir(shard_dir)

    workers = num_workers if num_workers > 0 else max((os.cpu_count() or 1) - 1, 1)
    workers = min(workers, len(labeled_images))
    print(f"[preprocess] Found {len(labeled_images)} images under {dataset_root}")
    print(f"[preprocess] Using {workers} worker thread(s)")
    print("[preprocess] Starting preprocessing batches now")

    tasks: List[Tuple[int, str, str, str, int]] = []
    for idx, (_, image_path) in enumerate(labeled_images):
        out_name = f"{idx}.png"
        tasks.append(
            (
                idx,
                str(image_path),
                str(images_out / out_name),
                str(cond_out / out_name),
                output_size,
            )
        )

    batch_size = max(64, min(512, math.ceil(len(tasks) / max(workers * 4, 1))))
    batches = [tasks[start : start + batch_size] for start in range(0, len(tasks), batch_size)]
    batch_jobs: List[Tuple[int, Sequence[Tuple[int, str, str, str, int]], str, int, int, int, Path]] = []
    for batch_idx, batch_tasks in enumerate(batches):
        shard_metadata_path = shard_dir / f"{batch_idx}.jsonl"
        batch_jobs.append(
            (
                batch_idx,
                batch_tasks,
                str(shard_metadata_path),
                output_size,
                line_thickness,
                joint_radius,
                hand_landmarker_model_path,
            )
        )

    total_written = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_batch, job): job[0] for job in batch_jobs}
        for future in as_completed(future_map):
            batch_idx, batch_written = future.result()
            total_written += batch_written
            print(f"[preprocess] batch {batch_idx + 1}/{len(batch_jobs)} wrote {batch_written} samples")

    with metadata_path.open("w", encoding="utf-8") as meta_fp:
        for batch_idx in range(len(batch_jobs)):
            shard_metadata_path = shard_dir / f"{batch_idx}.jsonl"
            if not shard_metadata_path.exists():
                continue
            with shard_metadata_path.open("r", encoding="utf-8") as shard_fp:
                for line in shard_fp:
                    if line.strip():
                        meta_fp.write(line)

    shutil.rmtree(shard_dir)
    print(f"[preprocess] Wrote {total_written} samples")

    if auto_caption:
        print("[preprocess] Running BLIP auto-captioning")
        caption_metadata_file(
            dataset_root=output_root,
            metadata_path=metadata_path,
            batch_size=caption_batch_size,
            limit=caption_limit,
            force=caption_force,
            model_name=caption_model_name,
        )

    print(f"[preprocess] Done. Output at: {output_root}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess HAGRID for ControlNet")
    parser.add_argument("--archive-path", type=Path, default=Path("data/archive/hagrid.zip"), help="Path to HAGRID zip")
    parser.add_argument("--extracted-root", type=Path, default=Path("data/raw/hagrid"), help="Extract destination")
    parser.add_argument("--output-root", type=Path, default=Path("data/hagrid_train"), help="Output dataset root")
    parser.add_argument("--limit", type=int, default=0, help="Max number of images to preprocess. 0 = all")
    parser.add_argument("--num-workers", type=int, default=0, help="CPU workers for preprocessing. 0 = auto")
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="Output square size")
    parser.add_argument("--line-thickness", type=int, default=2, help="Skeleton line thickness")
    parser.add_argument("--joint-radius", type=int, default=4, help="Skeleton joint radius")
    parser.add_argument("--disable-auto-caption", action="store_true", help="Skip BLIP captioning after preprocess")
    parser.add_argument("--caption-batch-size", type=int, default=256, help="BLIP caption batch size")
    parser.add_argument("--caption-limit", type=int, default=0, help="Max captions to generate. 0 = all")
    parser.add_argument("--caption-force", action="store_true", help="Overwrite existing captions")
    parser.add_argument(
        "--hand-landmarker-model-path",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to MediaPipe HandLandmarker .task model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_hagrid(
        archive_path=args.archive_path,
        extracted_root=args.extracted_root,
        output_root=args.output_root,
        output_size=args.output_size,
        line_thickness=args.line_thickness,
        joint_radius=args.joint_radius,
        limit=args.limit,
        num_workers=args.num_workers,
        auto_caption=not args.disable_auto_caption,
        caption_batch_size=args.caption_batch_size,
        caption_limit=args.caption_limit,
        caption_force=args.caption_force,
        caption_model_name="Salesforce/blip-image-captioning-base",
        hand_landmarker_model_path=args.hand_landmarker_model_path,
    )


if __name__ == "__main__":
    main()
