
import json
import os
import urllib.request
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

DEFAULT_FREIHAND_URL = (
	"https://lmb.informatik.uni-freiburg.de/data/freihand/"
	"FreiHAND_pub_v2.zip"
)

DEFAULT_OUTPUT_SIZE = 512

HAND_BONES = (
	(0, 1),
	(1, 2),
	(2, 3),
	(3, 4),
	(0, 5),
	(5, 6),
	(6, 7),
	(7, 8),
	(0, 9),
	(9, 10),
	(10, 11),
	(11, 12),
	(0, 13),
	(13, 14),
	(14, 15),
	(15, 16),
	(0, 17),
	(17, 18),
	(18, 19),
	(19, 20),
)

FINGER_COLORS = (
	(255, 64, 64),   # Thumb
	(255, 160, 64),  # Index
	(255, 220, 64),  # Middle
	(64, 200, 255),  # Ring
	(120, 120, 255), # Pinky
)

def ensure_dir(path: Path) -> None:
	"""Create directory if it doesn't exist"""
	path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
	"""Download file from URL with progress indicator"""
	ensure_dir(destination.parent)
	if destination.exists() and destination.stat().st_size > 0:
		print(f"[download] Archive already exists: {destination}")
		return

	print(f"[download] Downloading {url}")

	def progress(block_count: int, block_size: int, total_size: int) -> None:
		if total_size <= 0:
			return
		downloaded = block_count * block_size
		ratio = min(downloaded / total_size, 1.0)
		print(f"\r[download] {ratio:.1%} ({downloaded // (1024 * 1024)} MB)", end="")

	urllib.request.urlretrieve(url, destination, reporthook=progress)
	print("\n[download] Completed")


def extract_zip(archive_path: Path, extract_to: Path) -> None:
	"""Extract zip archive to destination directory"""
	ensure_dir(extract_to)
	print(f"[extract] Extracting {archive_path} -> {extract_to}")
	with zipfile.ZipFile(archive_path, "r") as zf:
		zf.extractall(extract_to)
	print("[extract] Completed")


def load_json(path: Path) -> Any:
	"""Load JSON file"""
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def project_xyz_to_uv(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
	"""Project 3D joint coordinates to 2D image plane using camera intrinsics"""
	homogeneous = (k @ xyz.T).T
	z = np.clip(homogeneous[:, 2:3], 1e-6, None)
	uv = homogeneous[:, :2] / z
	return uv


def draw_skeleton(joints_uv: np.ndarray, image_size: Tuple[int, int], line_thickness: int, joint_radius: int) -> np.ndarray:
	"""Draw hand skeleton visualization on image"""
	h, w = image_size
	canvas = np.zeros((h, w, 3), dtype=np.uint8)

	def finger_idx(joint_idx: int) -> int:
		if 1 <= joint_idx <= 4:
			return 0
		if 5 <= joint_idx <= 8:
			return 1
		if 9 <= joint_idx <= 12:
			return 2
		if 13 <= joint_idx <= 16:
			return 3
		return 4

	for j0, j1 in HAND_BONES:
		p0 = tuple(np.round(joints_uv[j0]).astype(int).tolist())
		p1 = tuple(np.round(joints_uv[j1]).astype(int).tolist())
		color = FINGER_COLORS[finger_idx(j1)]
		cv2.line(canvas, p0, p1, color, thickness=line_thickness, lineType=cv2.LINE_AA)

	for idx, point in enumerate(joints_uv):
		x, y = np.round(point).astype(int)
		color = (255, 255, 255) if idx == 0 else FINGER_COLORS[finger_idx(idx)]
		cv2.circle(canvas, (x, y), joint_radius, color, thickness=-1, lineType=cv2.LINE_AA)

	return canvas

def resize_pair(rgb: np.ndarray, control: np.ndarray, output_size: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Resize RGB and control images to target size"""
	rgb_resized = cv2.resize(rgb, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
	control_resized = cv2.resize(control, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
	return rgb_resized, control_resized

def sorted_image_files(image_dir: Path) -> List[Path]:
	"""Get sorted list of image files from directory"""
	files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
	files.sort(key=lambda p: p.name)
	return files


def process_one_sample(
	task: Tuple[int, str, List[List[float]], List[List[float]], str, str, int, int, int]
) -> Optional[Tuple[int, Dict[str, str]]]:
	"""Process single sample: project joints, draw skeleton, and save outputs"""
	idx, image_path_str, xyz_entry, k_entry, images_out_str, cond_out_str, output_size, line_thickness, joint_radius = task

	rgb_bgr = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
	if rgb_bgr is None:
		return None

	xyz = np.array(xyz_entry, dtype=np.float32)
	k = np.array(k_entry, dtype=np.float32)
	uv = project_xyz_to_uv(xyz, k)

	h, w = rgb_bgr.shape[:2]
	uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
	uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)

	control_bgr = draw_skeleton(
		joints_uv=uv,
		image_size=(h, w),
		line_thickness=line_thickness,
		joint_radius=joint_radius,
	)

	rgb_bgr, control_bgr = resize_pair(rgb_bgr, control_bgr, output_size)

	out_name = f"{idx:08d}.png"
	rgb_out_path = Path(images_out_str) / out_name
	cond_out_path = Path(cond_out_str) / out_name

	if not cv2.imwrite(str(rgb_out_path), rgb_bgr):
		return None
	if not cv2.imwrite(str(cond_out_path), control_bgr):
		return None

	row = {
		"file_name": str(Path("images") / out_name).replace("\\", "/"),
		"conditioning_image": str(Path("conditioning_images") / out_name).replace("\\", "/"),
		"caption": "a realistic human hand",
	}
	return idx, row

def preprocess_freihand(dataset_root: Path, 
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
						caption_model_name: str = "Salesforce/blip-image-captioning-base") -> None:
	"""Preprocess FreiHAND dataset with skeleton visualization and optional captioning"""

	rgb_dir = dataset_root / "training" / "rgb"
	xyz_path, k_path = dataset_root / "training_xyz.json", dataset_root / "training_K.json"

	xyz_all = load_json(xyz_path)
	k_all = load_json(k_path)
	image_paths = sorted_image_files(rgb_dir)

	sample_count = min(len(image_paths), len(xyz_all), len(k_all))
	
	if limit > 0:
		sample_count = min(sample_count, limit)

	images_out = output_root / "images"
	cond_out = output_root / "conditioning_images"
	ensure_dir(images_out)
	ensure_dir(cond_out)

	metadata_path = output_root / "metadata.jsonl"
	if metadata_path.exists():
		print(f"[preprocess] Existing preprocessed data found at: {output_root}. Skipping.")
		return

	workers = num_workers if num_workers > 0 else max((os.cpu_count() or 1) - 1, 1)
	print(f"[preprocess] Processing {sample_count} samples with {workers} workers")

	tasks: List[Tuple[int, str, List[List[float]], List[List[float]], str, str, int, int, int]] = []
	for idx in range(sample_count):
		tasks.append(
			(
				idx,
				str(image_paths[idx]),
				xyz_all[idx],
				k_all[idx],
				str(images_out),
				str(cond_out),
				output_size,
				line_thickness,
				joint_radius,
			)
		)

	rows_by_idx: Dict[int, Dict[str, str]] = {}
	processed = 0

	with ProcessPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(process_one_sample, task) for task in tasks]
		for future in as_completed(futures):
			result = future.result()
			processed += 1
			if result is not None:
				idx, row = result
				rows_by_idx[idx] = row

			if processed % 500 == 0 or processed == sample_count:
				print(f"[preprocess] {processed}/{sample_count}")

	with metadata_path.open("w", encoding="utf-8") as meta_fp:
		for idx in range(sample_count):
			row = rows_by_idx.get(idx)
			if row is not None:
				meta_fp.write(json.dumps(row) + "\n")

	if auto_caption:
		from scripts.caption_dataset import caption_metadata_file

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
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description="Preprocess FreiHAND for ControlNet")
	parser.add_argument("--limit", type=int, default=0, help="Max number of samples to preprocess. 0 = all")
	parser.add_argument("--num-workers", type=int, default=0, help="Worker count. 0 = auto")
	parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="Output square size")
	parser.add_argument("--line-thickness", type=int, default=2, help="Skeleton line thickness")
	parser.add_argument("--joint-radius", type=int, default=4, help="Skeleton joint radius")
	parser.add_argument("--disable-auto-caption", action="store_true", help="Skip BLIP captioning after preprocess")
	parser.add_argument("--caption-batch-size", type=int, default=128, help="BLIP caption batch size")
	parser.add_argument("--caption-limit", type=int, default=0, help="Max captions to generate. 0 = all")
	parser.add_argument("--caption-force", action="store_true", help="Overwrite existing captions")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	data_dir = Path("data")
	archives_dir = data_dir / "archives"
	dataset_root = data_dir / "raw"
	output_dir = data_dir / "freihand_controlnet"
	archive_path = archives_dir / "FreiHAND_pub_v2.zip"
	
	# download_file(DEFAULT_FREIHAND_URL, archive_path)
	# extract_zip(archive_path, dataset_root)
	preprocess_freihand(
		dataset_root=dataset_root,
		output_root=output_dir,
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
	)

if __name__ == "__main__":
	main()

