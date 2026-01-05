import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

from sports.common.team import TeamClassifier

from activity_tracker import ActivityTracker


def extract_player_crops(model: YOLO, video_path: str, stride: int = 30, max_crops: int = 120) -> list[np.ndarray]:
    print(f"[INFO] Extracting player crops from {video_path} (stride={stride})...")

    crops: list[np.ndarray] = []
    frame_generator = sv.get_video_frames_generator(video_path)

    for frame_idx, frame in enumerate(frame_generator):
        if frame_idx % stride != 0:
            continue

        result = model.predict(frame, conf=0.3, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == 2]

        for xyxy in player_detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    crops.append(crop)

        if len(crops) >= max_crops:
            break

    print(f"[INFO] Extracted {len(crops)} player crops")
    return crops


def main() -> None:
    repo_dir = Path(__file__).resolve().parent

    data_dir = repo_dir / "data" / "raw"
    models_dir = repo_dir / "models"
    outputs_dir = repo_dir / "outputs"
    reports_dir = repo_dir / "reports"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_video = os.environ.get("SOURCE_VIDEO_PATH", str(data_dir / "08fd33_0.mp4"))
    model_path = os.environ.get("MODEL_PATH", str(models_dir / "foatball350.pt"))
    target_video = os.environ.get("TARGET_VIDEO_PATH", str(outputs_dir / "annotated_output.mp4"))

    if not Path(source_video).exists():
        raise FileNotFoundError(f"SOURCE_VIDEO_PATH not found: {source_video}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {model_path}")

    print("=" * 60)
    print("FOOTBALL PLAYER TRACKING WITH TEAM CLASSIFICATION")
    print("=" * 60)

    print("[STEP 0] Loading YOLO model...")
    model = YOLO(model_path)

    print("[STEP 1] Extracting crops...")
    crops = extract_player_crops(model, source_video, stride=30)
    if not crops:
        raise RuntimeError("No player crops were extracted; check class IDs or confidence threshold.")

    print("[STEP 2] Training TeamClassifier...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if device == "cuda" else 4
    print(f"[INFO] Using device={device} (batch_size={batch_size})")

    team_classifier = TeamClassifier(device=device, batch_size=batch_size)
    team_classifier.fit(crops)

    print("[STEP 3] Setting up annotators + tracker...")
    team_palette = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493"])
    ellipse_player = sv.EllipseAnnotator(color=team_palette, thickness=2)
    label_player = sv.LabelAnnotator(
        color=team_palette,
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_scale=0.5,
        text_thickness=1,
        text_padding=3,
    )

    box_goalkeeper = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    label_goalkeeper = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.3)

    box_referee = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
    label_referee = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.3)

    triangle_ball = sv.TriangleAnnotator(color=sv.Color.YELLOW, base=25, height=20)

    tracker = sv.ByteTrack()

    print("[STEP 4] Processing video...")
    video_info = sv.VideoInfo.from_video_path(source_video)
    frame_generator = sv.get_video_frames_generator(source_video)

    activity_tracker = ActivityTracker(fps=video_info.fps)

    team_cache: dict[int, int] = {}
    last_update: dict[int, int] = {}
    TEAM_UPDATE_INTERVAL = 15

    labels: list[str] = []

    with sv.VideoSink(target_video, video_info=video_info) as sink:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Processing")):
            result = model.predict(frame, conf=0.2, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detection = detections[detections.class_id == 0]
            if len(ball_detection) > 0:
                ball_detection.xyxy = sv.pad_boxes(ball_detection.xyxy, px=5, py=5)

            ball_xyxy = None
            if len(ball_detection) > 0:
                try:
                    best_ball_idx = int(np.argmax(ball_detection.confidence))
                except Exception:
                    best_ball_idx = 0
                ball_xyxy = ball_detection.xyxy[best_ball_idx]

            player_detection = detections[detections.class_id == 2]
            goalkeeper_detection = detections[detections.class_id == 1]
            referee_detection = detections[detections.class_id == 3]

            labels = []

            # Defaults for frames without tracked players
            player_xyxy = np.empty((0, 4), dtype=float)
            player_tracker_ids = np.array([], dtype=object)
            player_team_ids = np.array([], dtype=int)

            if len(player_detection) > 0:
                player_detection = player_detection.with_nms(threshold=0.3, class_agnostic=True)
                player_detection = tracker.update_with_detections(player_detection)

                team_ids = [0] * len(player_detection)
                crops_to_process: list[np.ndarray] = []
                indices_to_process: list[int] = []

                for idx, (xyxy, tracker_id) in enumerate(zip(player_detection.xyxy, player_detection.tracker_id)):
                    if tracker_id is not None:
                        if tracker_id in team_cache and frame_idx - last_update.get(tracker_id, 0) < TEAM_UPDATE_INTERVAL:
                            team_ids[idx] = team_cache[tracker_id]
                            continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                            crops_to_process.append(crop)
                            indices_to_process.append(idx)

                if crops_to_process:
                    new_team_ids = team_classifier.predict(crops_to_process)
                    for idx, team_id in zip(indices_to_process, new_team_ids):
                        team_ids[idx] = int(team_id)
                        tracker_id = player_detection.tracker_id[idx]
                        if tracker_id is not None:
                            team_cache[int(tracker_id)] = int(team_id)
                            last_update[int(tracker_id)] = frame_idx

                player_detection.class_id = team_ids

                player_xyxy = np.asarray(player_detection.xyxy)
                player_tracker_ids = np.asarray(player_detection.tracker_id, dtype=object)
                player_team_ids = np.asarray(player_detection.class_id, dtype=int)

                for tracker_id, team_id in zip(player_detection.tracker_id, team_ids):
                    if tracker_id is not None:
                        labels.append(f"T{team_id+1}-{int(tracker_id)}")
                    else:
                        labels.append(f"T{team_id+1}")

            activity_tracker.update(
                frame_idx=frame_idx,
                player_xyxy=player_xyxy,
                player_tracker_ids=player_tracker_ids,
                player_team_ids=player_team_ids,
                ball_xyxy=ball_xyxy,
            )

            annotated = frame.copy()

            if len(player_detection) > 0:
                annotated = ellipse_player.annotate(scene=annotated, detections=player_detection)
                if labels and len(labels) == len(player_detection):
                    annotated = label_player.annotate(scene=annotated, detections=player_detection, labels=labels)

            if len(ball_detection) > 0:
                annotated = triangle_ball.annotate(scene=annotated, detections=ball_detection)

            if len(goalkeeper_detection) > 0:
                annotated = box_goalkeeper.annotate(scene=annotated, detections=goalkeeper_detection)
                gk_labels = [f"GK {conf:.2f}" for conf in goalkeeper_detection.confidence]
                annotated = label_goalkeeper.annotate(scene=annotated, detections=goalkeeper_detection, labels=gk_labels)

            if len(referee_detection) > 0:
                annotated = box_referee.annotate(scene=annotated, detections=referee_detection)
                ref_labels = [f"REF {conf:.2f}" for conf in referee_detection.confidence]
                annotated = label_referee.annotate(scene=annotated, detections=referee_detection, labels=ref_labels)

            sink.write_frame(annotated)

    print("=" * 60)
    print("✅ PROCESSING COMPLETE")
    print(f"Output saved to: {target_video}")

    report_prefix = Path(target_video).stem
    summary_csv, events_csv, team_csv, freq_csv = activity_tracker.write_reports(output_dir=reports_dir, prefix=report_prefix)
    print(f"Activity report saved to: {summary_csv}")
    print(f"Activity events saved to: {events_csv}")
    print(f"Team report saved to: {team_csv}")
    print(f"Pass frequency saved to: {freq_csv}")
    print("=" * 60)


if __name__ == "__main__":
    # Optional: suppress symlink warning on Windows if desired
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    main()
