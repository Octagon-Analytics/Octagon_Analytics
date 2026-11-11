"""YOLO pose-video utilities for the Footage Analysis page.

The measurements are screen-space (2D) summaries intended for exploration.
They are not validated biomechanical measurements and are not used by the
fight-outcome models.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b in degrees; returns NaN if any required keypoint is missing."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denominator, -1, 1))))


def _mean_points(points: np.ndarray, indices: list[int], confidence: np.ndarray) -> np.ndarray:
    selected = [points[i] for i in indices if i < len(confidence) and confidence[i] >= 0.35]
    return np.mean(selected, axis=0) if selected else np.array([np.nan, np.nan])


def analyze_fight_video(
    video_bytes: bytes,
    suffix: str,
    sample_fps: float = 2.0,
    max_seconds: int = 60,
    confidence: float = 0.45,
) -> dict:
    """Analyze a clip and return a compact annotated MP4 plus summary tables.

    YOLO's tracker supplies temporary anonymous IDs. IDs can swap when fighters
    overlap, so the output deliberately labels people as tracked subjects rather
    than assigning their names.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the video-analysis requirements before using this tab.") from exc

    with tempfile.TemporaryDirectory(prefix="octagon_pose_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / f"input{suffix or '.mp4'}"
        output_path = temp / "pose_summary.mp4"
        input_path.write_bytes(video_bytes)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("The uploaded file could not be read as a video. Try MP4 or MOV.")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / source_fps if source_fps else 0
        if width == 0 or height == 0:
            raise ValueError("The uploaded video has no readable frames.")

        frame_step = max(1, round(source_fps / sample_fps))
        max_frames = min(total_frames, int(max_seconds * source_fps))
        output_fps = source_fps / frame_step
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the annotated MP4 output.")

        pose_model = YOLO("yolo11n-pose.pt")
        previous: dict[int, tuple[float, np.ndarray]] = {}
        subject_rows: list[dict] = []
        frames_sampled = 0
        frames_with_person = 0
        people_detected = 0
        frame_index = 0

        while frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step:
                frame_index += 1
                continue

            time_seconds = frame_index / source_fps
            result = pose_model.track(
                frame, persist=True, classes=[0], conf=confidence, verbose=False
            )[0]
            annotated = result.plot()
            cv2.putText(
                annotated, f"Pose sample: {time_seconds:05.1f}s", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(annotated)
            frames_sampled += 1

            if result.keypoints is not None and len(result.keypoints):
                frames_with_person += 1
                points_all = result.keypoints.xyn.cpu().numpy()
                confidence_all = result.keypoints.conf.cpu().numpy()
                ids = (list(range(1, len(points_all) + 1)) if result.boxes.id is None
                       else result.boxes.id.int().cpu().tolist())
                for subject_id, points, keypoint_confidence in zip(ids, points_all, confidence_all):
                    people_detected += 1
                    wrists = _mean_points(points, [9, 10], keypoint_confidence)
                    def point(index):
                        return points[index] if keypoint_confidence[index] >= 0.35 else np.array([np.nan, np.nan])
                    wrist_speed = np.nan
                    if subject_id in previous and not np.any(np.isnan(wrists)):
                        previous_time, previous_wrists = previous[subject_id]
                        elapsed = time_seconds - previous_time
                        if elapsed > 0 and not np.any(np.isnan(previous_wrists)):
                            wrist_speed = float(np.linalg.norm(wrists - previous_wrists) / elapsed)
                    if not np.any(np.isnan(wrists)):
                        previous[subject_id] = (time_seconds, wrists)
                    subject_rows.append({
                        "Tracked subject": f"Subject {subject_id}", "time_seconds": time_seconds,
                        "pose_confidence": float(np.nanmean(keypoint_confidence)), "wrist_activity": wrist_speed,
                        "left_elbow_angle": _angle(point(5), point(7), point(9)),
                        "left_knee_angle": _angle(point(11), point(13), point(15)),
                    })
            frame_index += 1

        capture.release()
        writer.release()
        if not frames_sampled:
            raise ValueError("No frames could be sampled from this video.")

        observations = pd.DataFrame(subject_rows)
        if observations.empty:
            subjects = pd.DataFrame(columns=["Tracked subject", "Pose samples", "Mean pose confidence", "Mean wrist activity", "Mean left elbow angle", "Mean left knee angle"])
        else:
            subjects = observations.groupby("Tracked subject", as_index=False).agg(**{
                "Pose samples": ("time_seconds", "size"), "Mean pose confidence": ("pose_confidence", "mean"),
                "Mean wrist activity": ("wrist_activity", "mean"), "Mean left elbow angle": ("left_elbow_angle", "mean"),
                "Mean left knee angle": ("left_knee_angle", "mean"),
            })
        return {
            "video_bytes": output_path.read_bytes(), "subjects": subjects,
            "summary": {
                "Original clip duration": duration, "Duration analyzed": min(duration, max_seconds),
                "Pose samples": frames_sampled, "Frames with a detected person": frames_with_person,
                "Detection rate": frames_with_person / frames_sampled,
                "Mean people detected / sample": people_detected / frames_sampled,
            },
        }
