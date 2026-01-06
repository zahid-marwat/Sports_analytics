from __future__ import annotations

from pathlib import Path

import cv2


def main() -> None:
    video_path = Path("outputs") / "annotated_ui_test_v3.mp4"
    out_dir = Path("outputs") / "ui_test_v3_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    picks = [
        0,
        min(50, max(0, total - 1)),
        min(150, max(0, total - 1)),
        min(300, max(0, total - 1)),
        min(600, max(0, total - 1)),
    ]
    picks = sorted(set(p for p in picks if p >= 0))

    print(f"video={video_path} total_frames={total} picks={picks}")

    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"failed_read frame={idx}")
            continue

        out_path = out_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"wrote {out_path}")

    cap.release()


if __name__ == "__main__":
    main()
