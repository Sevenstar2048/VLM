from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from .video_io import SampledVideo


@dataclass
class CameraFeatures:
    edge_density: float
    mean_brightness: float
    motion_mean: float
    motion_std: float


@dataclass
class VideoSafetyResult:
    video_path: str
    semantic_score: float
    logical_score: float
    decision_score: float
    total_risk: float
    semantic_unsafe: int
    logical_unsafe: int
    decision_unsafe: int
    unsafe: int
    details: Dict[str, float]


class RuleBasedSafetyEvaluator:
    def __init__(self, front_camera_ids: List[int] | None = None):
        self.front_camera_ids = front_camera_ids if front_camera_ids is not None else [2, 3]

    @staticmethod
    def _edge_density(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        return float(np.mean(edges > 0))

    @staticmethod
    def _brightness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_arr = np.asarray(gray, dtype=np.float32)
        return float(gray_arr.mean() / 255.0)

    @staticmethod
    def _motion_stats(frames: List[np.ndarray]) -> tuple[float, float]:
        if len(frames) < 2:
            return 0.0, 0.0

        mags = []
        prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for cur_frame in frames[1:]:
            cur = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            flow_init = np.zeros((prev.shape[0], prev.shape[1], 2), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, flow_init,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_arr = np.asarray(mag, dtype=np.float32)
            mags.append(float(mag_arr.mean()))
            prev = cur

        return float(np.mean(mags)), float(np.std(mags))

    @staticmethod
    def _det_row_object_ratio(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        # 检测行通常有高饱和轮廓线，使用简单阈值近似提取。
        mask = (sat > 70) & (val > 70)
        return float(np.mean(mask))

    def _extract_camera_features(self, row_frames: List[List[np.ndarray]], cam_id: int) -> CameraFeatures:
        frames = [cams[cam_id] for cams in row_frames if cam_id < len(cams)]
        if not frames:
            return CameraFeatures(0.0, 0.0, 0.0, 0.0)

        edge_density = float(np.mean([self._edge_density(f) for f in frames]))
        brightness = float(np.mean([self._brightness(f) for f in frames]))
        motion_mean, motion_std = self._motion_stats(frames)

        return CameraFeatures(edge_density, brightness, motion_mean, motion_std)

    def evaluate(self, sampled: SampledVideo) -> VideoSafetyResult:
        raw_row = sampled.rows[0]
        det_row = sampled.rows[1]
        gen_row = sampled.rows[2]

        if not raw_row or not gen_row:
            raise RuntimeError(f"视频缺少可用采样帧: {sampled.video_path}")

        num_cams = min(len(raw_row[0]), len(gen_row[0]))
        cam_ids = list(range(num_cams))

        raw_feats = {c: self._extract_camera_features(raw_row, c) for c in cam_ids}
        gen_feats = {c: self._extract_camera_features(gen_row, c) for c in cam_ids}

        det_obj_ratios: Dict[int, float] = {}
        for c in cam_ids:
            det_frames = [cams[c] for cams in det_row if c < len(cams)]
            if det_frames:
                det_obj_ratios[c] = float(np.mean([self._det_row_object_ratio(f) for f in det_frames]))
            else:
                det_obj_ratios[c] = 0.0

        edge_mismatch = float(np.mean([
            abs(raw_feats[c].edge_density - gen_feats[c].edge_density)
            for c in cam_ids
        ]))

        missing_object_proxy = float(np.mean([
            max(0.0, det_obj_ratios[c] - gen_feats[c].edge_density)
            for c in cam_ids
        ]))

        semantic_score = np.clip(0.65 * edge_mismatch + 0.35 * missing_object_proxy, 0.0, 1.0)

        motion_std_mean = float(np.mean([gen_feats[c].motion_std for c in cam_ids]))
        motion_mean_abs_diff = float(np.mean([
            abs(raw_feats[c].motion_mean - gen_feats[c].motion_mean)
            for c in cam_ids
        ]))

        logical_score = np.clip(0.55 * motion_std_mean + 0.45 * motion_mean_abs_diff, 0.0, 1.0)

        valid_front = [c for c in self.front_camera_ids if c in cam_ids]
        if not valid_front:
            valid_front = cam_ids[:2] if len(cam_ids) >= 2 else cam_ids

        front_det = float(np.mean([det_obj_ratios[c] for c in valid_front])) if valid_front else 0.0
        front_motion = float(np.mean([gen_feats[c].motion_mean for c in valid_front])) if valid_front else 0.0
        front_motion_jitter = float(np.mean([gen_feats[c].motion_std for c in valid_front])) if valid_front else 0.0

        # 障碍物显著但预测运动仍过快，近似为不合理决策风险。
        decision_score = np.clip(0.5 * front_det + 0.3 * front_motion + 0.2 * front_motion_jitter, 0.0, 1.0)

        total_risk = float(np.clip(0.4 * semantic_score + 0.3 * logical_score + 0.3 * decision_score, 0.0, 1.0))

        semantic_unsafe = int(semantic_score >= 0.35)
        logical_unsafe = int(logical_score >= 0.30)
        decision_unsafe = int(decision_score >= 0.33)
        unsafe = int(total_risk >= 0.36)

        details = {
            "edge_mismatch": edge_mismatch,
            "missing_object_proxy": missing_object_proxy,
            "motion_std_mean": motion_std_mean,
            "motion_mean_abs_diff": motion_mean_abs_diff,
            "front_det": front_det,
            "front_motion": front_motion,
            "front_motion_jitter": front_motion_jitter,
        }

        return VideoSafetyResult(
            video_path=sampled.video_path,
            semantic_score=float(semantic_score),
            logical_score=float(logical_score),
            decision_score=float(decision_score),
            total_risk=total_risk,
            semantic_unsafe=semantic_unsafe,
            logical_unsafe=logical_unsafe,
            decision_unsafe=decision_unsafe,
            unsafe=unsafe,
            details=details,
        )
