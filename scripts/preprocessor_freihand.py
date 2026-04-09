
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

DEFAULT_FREIHAND_URL = (
	"https://lmb.informatik.uni-freiburg.de/data/freihand/"
	"FreiHAND_pub_v2.zip"
)

DEFAULT_OUTPUT_SIZE = 512

HAND_BONES: Tuple[Tuple[int, int], ...] = (
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

FINGER_COLORS: Tuple[Tuple[int, int, int], ...] = (
	(255, 64, 64),   # Thumb
	(255, 160, 64),  # Index
	(255, 220, 64),  # Middle
	(64, 200, 255),  # Ring
	(120, 120, 255), # Pinky
)

def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
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
	ensure_dir(extract_to)
	print(f"[extract] Extracting {archive_path} -> {extract_to}")
	with zipfile.ZipFile(archive_path, "r") as zf:
		zf.extractall(extract_to)
	print("[extract] Completed")


def load_json(path: Path):
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def project_xyz_to_uv(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
	homogeneous = (k @ xyz.T).T
	z = np.clip(homogeneous[:, 2:3], 1e-6, None)
	uv = homogeneous[:, :2] / z
	return uv


def draw_skeleton(joints_uv: np.ndarray, image_size: Tuple[int, int], line_thickness: int, joint_radius: int) -> np.ndarray:
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
	rgb_resized = cv2.resize(rgb, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
	control_resized = cv2.resize(control, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
	return rgb_resized, control_resized

def sorted_image_files(image_dir: Path) -> List[Path]:
	files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
	files.sort(key=lambda p: p.name)
	return files

def preprocess_freihand(dataset_root: Path, 
						output_root: Path, 
						output_size: int, 
						line_thickness: int, 
						joint_radius: int, 
						limit: int) -> None:

	rgb_dir = dataset_root / "training" / "rgb"
	xyz_path, k_path = dataset_root / "training_xyz.json", dataset_root / "training_K.json"

	xyz_all = load_json(xyz_path)
	k_all = load_json(k_path)
	image_paths = sorted_image_files(rgb_dir)

	sample_count = 10000

	images_out = output_root / "images"
	cond_out = output_root / "conditioning_images"
	ensure_dir(images_out)
	ensure_dir(cond_out)

	metadata_path = output_root / "metadata.jsonl"
	print(f"[preprocess] Processing {sample_count} samples")

	with metadata_path.open("w", encoding="utf-8") as meta_fp:
		
		for idx in range(sample_count):
			image_path = image_paths[idx]
			rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
			
			if rgb_bgr is None:
				continue

			xyz = np.array(xyz_all[idx], dtype=np.float32)
			k = np.array(k_all[idx], dtype=np.float32)
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
			rgb_out_path = images_out / out_name
			cond_out_path = cond_out / out_name

			cv2.imwrite(str(rgb_out_path), rgb_bgr)
			cv2.imwrite(str(cond_out_path), control_bgr)

			row = {
				"file_name": str(Path("images") / out_name).replace("\\", "/"),
				"conditioning_image": str(Path("conditioning_images") / out_name).replace("\\", "/"),
				"caption": "a realistic human hand"
			}

			meta_fp.write(json.dumps(row) + "\n")

			if (idx + 1) % 500 == 0 or (idx + 1) == sample_count:
				print(f"[preprocess] {idx + 1}/{sample_count}")

	print(f"[preprocess] Done. Output at: {output_root}")

def main() -> None:
	
	data_dir = Path("data")
	archives_dir = data_dir / "archives"
	dataset_root = data_dir / "raw"
	output_dir = data_dir / "freihand_controlnet"
	archive_path = archives_dir / "FreiHAND_pub_v2.zip"
	
	download_file(DEFAULT_FREIHAND_URL, archive_path)
	extract_zip(archive_path, dataset_root)
	preprocess_freihand(
		dataset_root=dataset_root,
		output_root=output_dir,
		output_size=DEFAULT_OUTPUT_SIZE, # Output square image size
		line_thickness=4, # Skeleton line thickness
		joint_radius=6, # Skeleton joint circle radius 
		limit=0 # Max num samples to preprocess, 0 = all
	)

if __name__ == "__main__":
	main()

