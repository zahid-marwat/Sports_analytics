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


class BallHeatmap:
    def __init__(self, *, grid_w: int = 64, grid_h: int = 36) -> None:
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self._hist = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def update(self, *, x_norm: float, y_norm: float) -> None:
        x = float(np.clip(x_norm, 0.0, 1.0))
        y = float(np.clip(y_norm, 0.0, 1.0))
        xi = int(np.clip(int(x * self.grid_w), 0, self.grid_w - 1))
        yi = int(np.clip(int(y * self.grid_h), 0, self.grid_h - 1))
        self._hist[yi, xi] += 1.0

    def render(self) -> np.ndarray:
        if float(self._hist.max()) <= 0.0:
            return np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        out = cv2.normalize(self._hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return out.astype(np.uint8)


def draw_ball_heatmap(
    frame: np.ndarray,
    *,
    heatmap_gray: np.ndarray,
    team_markers_norm: list[tuple[float, float, int]] | None = None,
    ball_marker_norm: tuple[float, float] | None = None,
    size: tuple[int, int] = (220, 130),
    pad: int = 12,
    alpha: float = 0.75,
) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    box_w, box_h = int(size[0]), int(size[1])

    x2 = w - pad
    y1 = pad
    x1 = max(pad, x2 - box_w)
    y2 = min(h - pad, y1 + box_h)

    if x2 <= x1 or y2 <= y1:
        return frame

    hm = cv2.resize(heatmap_gray, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    overlay = frame.copy()
    overlay[y1:y2, x1:x2] = hm_color
    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    # Small markers on top of the heatmap (keep tiny so they don't obscure the heatmap).
    # Colors are in BGR.
    # Team 1: red, Team 2: light green. Ball: white.
    if team_markers_norm:
        box_w_px = max(1, x2 - x1)
        box_h_px = max(1, y2 - y1)
        for x_norm, y_norm, team_id in team_markers_norm:
            try:
                xn = float(np.clip(float(x_norm), 0.0, 1.0))
                yn = float(np.clip(float(y_norm), 0.0, 1.0))
                tid = int(team_id)
            except Exception:
                continue

            px = int(x1 + xn * box_w_px)
            py = int(y1 + yn * box_h_px)
            # Light green (BGR) to avoid blending with the heatmap background.
            color = (0, 0, 255) if tid == 1 else (144, 238, 144)
            cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)

    if ball_marker_norm is not None:
        try:
            xn = float(np.clip(float(ball_marker_norm[0]), 0.0, 1.0))
            yn = float(np.clip(float(ball_marker_norm[1]), 0.0, 1.0))
            box_w_px = max(1, x2 - x1)
            box_h_px = max(1, y2 - y1)
            px = int(x1 + xn * box_w_px)
            py = int(y1 + yn * box_h_px)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
        except Exception:
            pass

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return frame


def _draw_football_icon(frame: np.ndarray, *, center: tuple[int, int], r: int = 7) -> None:
    # Simple drawn icon (white ball w/ black outline + center dot)
    cx, cy = int(center[0]), int(center[1])
    r = int(max(3, r))
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(1, r // 3), (0, 0, 0), -1, cv2.LINE_AA)


def draw_team_info_table(
    frame: np.ndarray,
    *,
    rows: list[dict[str, float | int]],
    possession_team_id: int | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if frame is None or frame.size == 0:
        return frame, (0, 0, 0, 0)

    # Normalize rows into team1/team2 buckets (team_id is 1-based here)
    by_team: dict[int, dict[str, float | int]] = {}
    for r in rows:
        try:
            tid = int(r.get("team_id", 0))
        except Exception:
            continue
        by_team[tid] = r

    t1 = by_team.get(1, {"team_id": 1, "passes": 0, "possession_pct": 0.0})
    t2 = by_team.get(2, {"team_id": 2, "passes": 0, "possession_pct": 0.0})

    t1_passes = int(t1.get("passes", 0))
    t2_passes = int(t2.get("passes", 0))
    t1_pos = float(t1.get("possession_pct", 0.0))
    t2_pos = float(t2.get("possession_pct", 0.0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    pad = 8
    icon_slot_w = 22

    # Table content
    headers = ["", "Team 1", "Team 2"]
    row_labels = ["Passes", "Possession"]
    values = [
        [str(t1_passes), str(t2_passes)],
        [f"{t1_pos:.1f}%", f"{t2_pos:.1f}%"],
    ]

    # Measure column widths
    def text_w(txt: str) -> int:
        (w, _h), _b = cv2.getTextSize(txt, font, font_scale, thickness)
        return int(w)

    col_w0 = max(text_w(headers[0]), *(text_w(lbl) for lbl in row_labels)) + pad * 2
    col_w1 = max(text_w(headers[1]) + icon_slot_w, *(text_w(v[0]) for v in values)) + pad * 2
    col_w2 = max(text_w(headers[2]) + icon_slot_w, *(text_w(v[1]) for v in values)) + pad * 2

    # Row heights
    (_w, base_h), base_b = cv2.getTextSize("Hg", font, font_scale, thickness)
    row_h = int(base_h + base_b + pad)

    x, y = 12, 12
    table_w = int(col_w0 + col_w1 + col_w2)
    table_h = int(row_h * (1 + len(row_labels)))

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + table_w, y + table_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Grid lines
    x1 = x + col_w0
    x2 = x + col_w0 + col_w1
    cv2.line(frame, (x1, y), (x1, y + table_h), (255, 255, 255), 1)
    cv2.line(frame, (x2, y), (x2, y + table_h), (255, 255, 255), 1)
    cv2.line(frame, (x, y + row_h), (x + table_w, y + row_h), (255, 255, 255), 1)
    cv2.line(frame, (x, y + row_h * 2), (x + table_w, y + row_h * 2), (255, 255, 255), 1)
    cv2.rectangle(frame, (x, y), (x + table_w, y + table_h), (255, 255, 255), 1)

    # Header text
    header_y = y + int(row_h * 0.70)
    cv2.putText(
        frame,
        headers[1],
        (x + col_w0 + pad + icon_slot_w, header_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        headers[2],
        (x + col_w0 + col_w1 + pad + icon_slot_w, header_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness + 1,
        cv2.LINE_AA,
    )

    # Possession icon near the team in possession
    if possession_team_id in (1, 2):
        col_x = x + col_w0 if possession_team_id == 1 else x + col_w0 + col_w1
        icon_center = (int(col_x + pad + (icon_slot_w // 2)), int(y + row_h // 2))
        _draw_football_icon(frame, center=icon_center, r=7)

    # Rows
    for r_i, label in enumerate(row_labels):
        cy = y + row_h * (r_i + 1) + int(row_h * 0.70)
        cv2.putText(frame, label, (x + pad, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        v1 = values[r_i][0]
        v2 = values[r_i][1]
        cv2.putText(frame, v1, (x + col_w0 + pad, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, v2, (x + col_w0 + col_w1 + pad, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame, (x, y, table_w, table_h)


def draw_last_activity_text(
    frame: np.ndarray,
    *,
    text: str | None,
    max_chars: int = 72,
    bottom_margin: int = 26,
) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    if not text:
        return frame

    # Basic wrapping into up to two lines
    line1 = text
    line2 = ""
    if len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        line1 = text[:cut].rstrip()
        line2 = text[cut:].lstrip()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    pad = 8

    lines = [line1] + ([line2] if line2 else [])
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]
    max_w = max(int(s[0][0]) for s in sizes)
    line_h = max(int(s[0][1] + s[1]) for s in sizes)

    box_w = max_w + pad * 2
    box_h = line_h * len(lines) + pad * 2

    h, w = frame.shape[:2]

    x = int(max(0, (w - box_w) // 2))
    y = int(max(0, h - bottom_margin - box_h))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cur_y = y + pad + line_h
    for t in lines:
        cv2.putText(frame, t, (x + pad, cur_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cur_y += line_h

    return frame


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
    target_video_env = os.environ.get("TARGET_VIDEO_PATH")
    if target_video_env:
        tv = Path(target_video_env)
        # If user only provides a filename, keep outputs/ as the default folder.
        if tv.parent == Path("."):
            target_video_path = outputs_dir / tv.name
        else:
            target_video_path = tv if tv.is_absolute() else (repo_dir / tv)
    else:
        target_video_path = outputs_dir / "annotated_output.mp4"

    target_video_path = target_video_path.resolve()
    target_video_path.parent.mkdir(parents=True, exist_ok=True)
    target_video = str(target_video_path)

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

    ball_heatmap = BallHeatmap(grid_w=64, grid_h=36)

    team_cache: dict[int, int] = {}
    last_update: dict[int, int] = {}
    TEAM_UPDATE_INTERVAL = 15

    labels: list[str] = []

    # Keep a stable guess of which team is defending each side.
    # Updated opportunistically from player positions.
    left_side_team: int | None = None
    right_side_team: int | None = None

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

                # Update cumulative ball heatmap (normalized by frame size)
                bx = float((float(ball_xyxy[0]) + float(ball_xyxy[2])) / 2.0)
                by = float((float(ball_xyxy[1]) + float(ball_xyxy[3])) / 2.0)
                fw = float(max(frame.shape[1], 1))
                fh = float(max(frame.shape[0], 1))
                ball_heatmap.update(x_norm=bx / fw, y_norm=by / fh)

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

                # Update side->team mapping from current frame player distribution.
                # Heuristic: whichever team has more players on the left half is the left-side team.
                frame_mid_x = float(frame.shape[1]) / 2.0
                centers_x = (player_xyxy[:, 0] + player_xyxy[:, 2]) / 2.0
                left_mask = centers_x < frame_mid_x
                right_mask = ~left_mask
                if np.any(left_mask):
                    left_counts = np.bincount(player_team_ids[left_mask], minlength=2)
                    if left_counts.sum() > 0 and left_counts[0] != left_counts[1]:
                        left_side_team = int(np.argmax(left_counts))
                if np.any(right_mask):
                    right_counts = np.bincount(player_team_ids[right_mask], minlength=2)
                    if right_counts.sum() > 0 and right_counts[0] != right_counts[1]:
                        right_side_team = int(np.argmax(right_counts))

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
                # Assign goalkeeper team using field-side heuristic:
                # If GK is on the left, assign the team that has more players on the left (and vice versa).
                gk_labels: list[str] = []
                frame_mid_x = float(frame.shape[1]) / 2.0
                for xyxy, conf in zip(goalkeeper_detection.xyxy, goalkeeper_detection.confidence):
                    gx = float((float(xyxy[0]) + float(xyxy[2])) / 2.0)
                    side = "left" if gx < frame_mid_x else "right"
                    team_guess = left_side_team if side == "left" else right_side_team
                    if team_guess is None:
                        gk_labels.append(f"GK {conf:.2f}")
                    else:
                        gk_labels.append(f"GK T{int(team_guess) + 1} {conf:.2f}")
                annotated = label_goalkeeper.annotate(scene=annotated, detections=goalkeeper_detection, labels=gk_labels)

            if len(referee_detection) > 0:
                annotated = box_referee.annotate(scene=annotated, detections=referee_detection)
                ref_labels = [f"REF {conf:.2f}" for conf in referee_detection.confidence]
                annotated = label_referee.annotate(scene=annotated, detections=referee_detection, labels=ref_labels)

            # Overlay team possession/pass summary up to this frame
            table_rows = activity_tracker.get_team_snapshot()
            possession_team_id = activity_tracker.get_current_possession_team_id()
            annotated, (tx, ty, tw, th) = draw_team_info_table(
                annotated,
                rows=table_rows,
                possession_team_id=possession_team_id,
            )

            # Overlay last activity text (subtitle-like)
            last_text = activity_tracker.get_last_activity_text()
            if last_text:
                annotated = draw_last_activity_text(
                    annotated,
                    text=last_text,
                )

            # Overlay mini ball heatmap
            heatmap_team_markers: list[tuple[float, float, int]] = []
            if player_xyxy is not None and len(player_xyxy) > 0:
                fw = float(max(frame.shape[1], 1))
                fh = float(max(frame.shape[0], 1))
                centers = (player_xyxy[:, 0:2] + player_xyxy[:, 2:4]) / 2.0
                for (cx, cy), tid0 in zip(centers, player_team_ids):
                    try:
                        team_id_1based = int(tid0) + 1
                    except Exception:
                        continue
                    heatmap_team_markers.append((float(cx) / fw, float(cy) / fh, team_id_1based))

            heatmap_ball_marker: tuple[float, float] | None = None
            if ball_xyxy is not None:
                bx = float((float(ball_xyxy[0]) + float(ball_xyxy[2])) / 2.0)
                by = float((float(ball_xyxy[1]) + float(ball_xyxy[3])) / 2.0)
                fw = float(max(frame.shape[1], 1))
                fh = float(max(frame.shape[0], 1))
                heatmap_ball_marker = (bx / fw, by / fh)

            annotated = draw_ball_heatmap(
                annotated,
                heatmap_gray=ball_heatmap.render(),
                team_markers_norm=heatmap_team_markers,
                ball_marker_norm=heatmap_ball_marker,
            )

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
