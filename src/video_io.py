from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class SampledVideo:
    video_path: str
    fps: float
    total_frames: int
    sampled_indices: List[int]
    # 结构: rows[row_idx][frame_idx][cam_idx] -> np.ndarray
    rows: Dict[int, List[List[np.ndarray]]]


class MultiViewGridParser:
    def __init__(self, rows: int = 3, cols: int = 6):
        self.rows = rows
        self.cols = cols

    def split_frame(self, frame: np.ndarray) -> List[List[np.ndarray]]:
        h, w = frame.shape[:2]
        cell_h = h // self.rows
        cell_w = w // self.cols

        grid: List[List[np.ndarray]] = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h if r < self.rows - 1 else h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w if c < self.cols - 1 else w
                row_cells.append(frame[y1:y2, x1:x2])
            grid.append(row_cells)
        return grid

    def sample_video(self, video_path: str, max_frames: int = 64) -> SampledVideo:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total_frames <= 0:
            indices = []
        elif total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, max_frames).astype(int).tolist()

        rows: Dict[int, List[List[np.ndarray]]] = {0: [], 1: [], 2: []}
        target_set = set(indices)

        current_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if current_idx in target_set:
                grid = self.split_frame(frame)
                for r in range(min(3, len(grid))):
                    rows[r].append(grid[r])
            current_idx += 1

        cap.release()

        return SampledVideo(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            sampled_indices=indices,
            rows=rows,
        )


def list_videos(data_dir: str) -> List[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    videos: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(str(p))
    videos.sort()
    return videos


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_keyframes(sampled: SampledVideo, out_dir: str, max_export: int = 8) -> Dict[str, List[str]]:
    ensure_dir(out_dir)
    exported: Dict[str, List[str]] = {"raw": [], "det": [], "gen": []}

    count = len(sampled.rows[0])
    if count == 0:
        return exported

    chosen = np.linspace(0, count - 1, min(max_export, count)).astype(int).tolist()

    for i, frame_idx in enumerate(chosen):
        for row_id, row_name in [(0, "raw"), (1, "det"), (2, "gen")]:
            cams = sampled.rows[row_id][frame_idx]
            # 每行由6个固定方位镜头横向拼接，给每个子画面标注摄像头序号，降低LLM误读概率。
            cell_h = cams[0].shape[0]
            cell_w = cams[0].shape[1]
            annotated: List[np.ndarray] = []
            for cam_id, cam in enumerate(cams):
                panel = cam.copy()
                cv2.putText(
                    panel,
                    f"cam{cam_id}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                annotated.append(panel)

            canvas = np.hstack(annotated)

            # 画垂直分隔线，明确6个分屏边界。
            for x in range(cell_w, cell_w * len(cams), cell_w):
                cv2.line(canvas, (x, 0), (x, cell_h - 1), (255, 255, 255), 2)

            cv2.putText(
                canvas,
                f"row={row_name} t={frame_idx}",
                (8, cell_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out_path = str(Path(out_dir) / f"{Path(sampled.video_path).stem}_{row_name}_{i:02d}.jpg")
            cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            exported[row_name].append(out_path)

    return exported
