from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


EventType = Literal["pass", "receive", "intercept", "possession_change"]


@dataclass
class PlayerStats:
    team_id: int
    player_id: int
    passes: int = 0
    received_passes: int = 0
    intercepts: int = 0
    possession_frames: int = 0


@dataclass
class ActivityEvent:
    frame_idx: int
    event: EventType
    from_team: int | None
    from_player: int | None
    to_team: int | None
    to_player: int | None


class ActivityTracker:
    """Heuristic ball-possession tracker to infer passes/receives/intercepts.

    Logic (simple + robust):
    - When the ball is detected, assign it to the nearest tracked player within a pixel threshold.
    - Require the same player to be nearest for a few consecutive frames (stable_frames) before confirming.
    - When confirmed possession changes:
        - same team: from_player += pass, to_player += received_pass
        - other team: to_player += intercept

    This is heuristic and depends on the ball detector quality.
    """

    def __init__(
        self,
        *,
        fps: float,
        max_assign_dist_px: float = 85.0,
        stable_frames: int = 3,
        max_ball_gap_frames: int = 12,
        min_event_gap_frames: int = 5,
        pass_freq_bin_seconds: float = 5.0,
        intercept_confirm_seconds: float = 1.0,
    ) -> None:
        self.fps = float(fps)
        self.max_assign_dist_px = float(max_assign_dist_px)
        self.stable_frames = int(stable_frames)
        self.max_ball_gap_frames = int(max_ball_gap_frames)
        self.min_event_gap_frames = int(min_event_gap_frames)
        self.pass_freq_bin_seconds = float(pass_freq_bin_seconds)
        self.intercept_confirm_seconds = float(intercept_confirm_seconds)
        self._min_intercept_frames = max(1, int(round(self.fps * self.intercept_confirm_seconds)))

        self._stats: dict[int, PlayerStats] = {}
        self._events: list[ActivityEvent] = []

        self._player_team: dict[int, int] = {}

        self._current_owner: int | None = None
        self._current_team: int | None = None
        self._candidate_owner: int | None = None
        self._candidate_count: int = 0

        self._last_ball_frame: int | None = None
        self._last_event_frame: int = -10_000

        self._team_possession_frames: dict[int, int] = {}
        self._team_passes: dict[int, int] = {}
        self._team_intercepts: dict[int, int] = {}

        # (team_id, frame_idx) for each pass event
        self._pass_events: list[tuple[int, int]] = []

        # Human-readable last activity for video overlay
        self._last_activity_text: str | None = None

        # Pending cross-team possession change (used to confirm interceptions).
        # We only count an intercept if the other team keeps possession for >= _min_intercept_frames.
        self._pending_cross_owner: int | None = None
        self._pending_cross_team: int | None = None
        self._pending_cross_start_frame: int | None = None
        self._pending_old_owner: int | None = None
        self._pending_old_team: int | None = None

    def _clear_pending_cross(self) -> None:
        self._pending_cross_owner = None
        self._pending_cross_team = None
        self._pending_cross_start_frame = None
        self._pending_old_owner = None
        self._pending_old_team = None

    @staticmethod
    def _center(xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = xyxy.astype(float)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

    def _get_or_create_stats(self, player_id: int, team_id: int) -> PlayerStats:
        existing = self._stats.get(player_id)
        if existing is not None:
            if existing.team_id != team_id:
                existing.team_id = team_id
            return existing

        stats = PlayerStats(team_id=team_id, player_id=player_id)
        self._stats[player_id] = stats
        return stats

    def _count_team_possession_frame(self) -> None:
        if self._current_team is None:
            return
        self._team_possession_frames[self._current_team] = self._team_possession_frames.get(self._current_team, 0) + 1

    def update(
        self,
        *,
        frame_idx: int,
        player_xyxy: np.ndarray,
        player_tracker_ids: np.ndarray,
        player_team_ids: np.ndarray,
        ball_xyxy: np.ndarray | None,
    ) -> None:
        frame_idx = int(frame_idx)

        # If we already know which team has possession, keep counting it.
        # Possession persists unless we confirm a new possessing player/team.
        self._count_team_possession_frame()

        # Update latest known team for each tracked player
        if player_tracker_ids is not None and len(player_tracker_ids) > 0:
            for tid, team in zip(player_tracker_ids, player_team_ids):
                if tid is None:
                    continue
                self._player_team[int(tid)] = int(team)

        if ball_xyxy is None:
            # Ball missing/undetected (common during passes/occlusions).
            # Do not emit an out-of-play event; just keep the last known possession.
            return

        self._last_ball_frame = frame_idx

        if player_xyxy is None or len(player_xyxy) == 0:
            return

        # Find nearest tracked player to the ball.
        ball_c = self._center(np.asarray(ball_xyxy))

        # Some detections can have tracker_id None; ignore those for ownership.
        valid = np.array([tid is not None for tid in player_tracker_ids], dtype=bool)
        if not np.any(valid):
            return

        xyxy_valid = np.asarray(player_xyxy)[valid]
        tid_valid = np.asarray(player_tracker_ids)[valid]
        team_valid = np.asarray(player_team_ids)[valid]

        player_centers = (xyxy_valid[:, 0:2] + xyxy_valid[:, 2:4]) / 2.0
        dists = np.linalg.norm(player_centers - ball_c[None, :], axis=1)

        best_i = int(np.argmin(dists))
        best_dist = float(dists[best_i])
        if best_dist > self.max_assign_dist_px:
            return

        nearest_id = int(tid_valid[best_i])
        nearest_team = int(team_valid[best_i])
        self._get_or_create_stats(nearest_id, nearest_team)

        # Stability smoothing
        if self._candidate_owner == nearest_id:
            self._candidate_count += 1
        else:
            self._candidate_owner = nearest_id
            self._candidate_count = 1

        if self._candidate_count < self.stable_frames:
            return

        # Confirmed ownership
        new_owner = nearest_id
        old_owner = self._current_owner
        if old_owner == new_owner:
            # still in possession
            if self._pending_cross_owner is not None:
                self._clear_pending_cross()
            team = self._player_team.get(new_owner, nearest_team)
            self._get_or_create_stats(new_owner, int(team)).possession_frames += 1
            # Ensure current team is set (first-time assignment)
            if self._current_team is None:
                self._current_team = int(team)
            return

        old_team = self._player_team.get(old_owner) if old_owner is not None else None
        new_team = self._player_team.get(new_owner, nearest_team)

        # If we don't have a previous owner/team yet, just initialize possession.
        if old_owner is None or old_team is None:
            self._clear_pending_cross()
            self._current_owner = new_owner
            self._current_team = int(new_team)
            self._last_event_frame = frame_idx
            self._get_or_create_stats(new_owner, int(new_team)).possession_frames += 1
            return

        # If we are waiting to confirm an interception and the nearest owner changes,
        # cancel the pending intercept.
        if self._pending_cross_owner is not None and int(new_owner) != int(self._pending_cross_owner):
            self._clear_pending_cross()

        # Same-team changes are immediate passes.
        if int(old_team) == int(new_team):
            self._clear_pending_cross()

            self._events.append(
                ActivityEvent(
                    frame_idx=frame_idx,
                    event="possession_change",
                    from_team=int(old_team),
                    from_player=old_owner,
                    to_team=int(new_team),
                    to_player=new_owner,
                )
            )

            self._get_or_create_stats(old_owner, int(old_team)).passes += 1
            self._get_or_create_stats(new_owner, int(new_team)).received_passes += 1

            self._team_passes[int(new_team)] = self._team_passes.get(int(new_team), 0) + 1
            self._pass_events.append((int(new_team), frame_idx))

            self._events.append(
                ActivityEvent(
                    frame_idx=frame_idx,
                    event="pass",
                    from_team=int(old_team),
                    from_player=old_owner,
                    to_team=int(new_team),
                    to_player=new_owner,
                )
            )
            self._last_activity_text = f"Team {int(new_team) + 1}: Player {int(old_owner)} passes to Player {int(new_owner)}"
            self._events.append(
                ActivityEvent(
                    frame_idx=frame_idx,
                    event="receive",
                    from_team=int(old_team),
                    from_player=old_owner,
                    to_team=int(new_team),
                    to_player=new_owner,
                )
            )

            self._current_owner = new_owner
            self._current_team = int(new_team)
            self._last_event_frame = frame_idx
            self._get_or_create_stats(new_owner, int(new_team)).possession_frames += 1
            return

        # Cross-team change: treat as a potential interception, but only confirm it
        # if the new team keeps possession for >= 1 second.
        if self._pending_cross_owner is None:
            self._pending_cross_owner = int(new_owner)
            self._pending_cross_team = int(new_team)
            self._pending_cross_start_frame = int(frame_idx)
            self._pending_old_owner = int(old_owner)
            self._pending_old_team = int(old_team)
            return

        # Continue pending confirmation (only if still the same new owner/team).
        if int(new_owner) != int(self._pending_cross_owner) or int(new_team) != int(self._pending_cross_team):
            self._clear_pending_cross()
            return

        assert self._pending_cross_start_frame is not None
        held_frames = int(frame_idx) - int(self._pending_cross_start_frame) + 1
        if held_frames < self._min_intercept_frames:
            return

        # Confirm interception now.
        pending_old_owner = self._pending_old_owner
        pending_old_team = self._pending_old_team
        self._clear_pending_cross()

        self._events.append(
            ActivityEvent(
                frame_idx=frame_idx,
                event="possession_change",
                from_team=int(pending_old_team) if pending_old_team is not None else None,
                from_player=int(pending_old_owner) if pending_old_owner is not None else None,
                to_team=int(new_team),
                to_player=new_owner,
            )
        )

        if pending_old_owner is not None and pending_old_team is not None:
            self._get_or_create_stats(new_owner, int(new_team)).intercepts += 1
            self._team_intercepts[int(new_team)] = self._team_intercepts.get(int(new_team), 0) + 1
            self._events.append(
                ActivityEvent(
                    frame_idx=frame_idx,
                    event="intercept",
                    from_team=int(pending_old_team),
                    from_player=int(pending_old_owner),
                    to_team=int(new_team),
                    to_player=new_owner,
                )
            )
            self._last_activity_text = (
                f"Team {int(new_team) + 1}: Player {int(new_owner)} intercepts ball from "
                f"Team {int(pending_old_team) + 1}: Player {int(pending_old_owner)}"
            )

        self._current_owner = new_owner
        self._current_team = int(new_team)
        self._last_event_frame = frame_idx
        self._get_or_create_stats(new_owner, int(new_team)).possession_frames += 1

    def write_reports(self, *, output_dir: str | Path, prefix: str) -> tuple[Path, Path, Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / f"{prefix}_activity_report.csv"
        events_path = output_dir / f"{prefix}_activity_events.csv"
        team_path = output_dir / f"{prefix}_team_report.csv"
        freq_path = output_dir / f"{prefix}_pass_frequency.csv"

        # Summary per player
        rows = sorted(self._stats.values(), key=lambda s: (s.team_id, s.player_id))
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "team_id",
                    "player_id",
                    "passes",
                    "received_passes",
                    "intercepts",
                    "possession_frames",
                    "possession_seconds",
                ],
            )
            writer.writeheader()
            for s in rows:
                writer.writerow(
                    {
                        "team_id": int(s.team_id) + 1,
                        "player_id": int(s.player_id),
                        "passes": int(s.passes),
                        "received_passes": int(s.received_passes),
                        "intercepts": int(s.intercepts),
                        "possession_frames": int(s.possession_frames),
                        "possession_seconds": round(float(s.possession_frames) / max(self.fps, 1e-6), 3),
                    }
                )

        # Event log (trace)
        with events_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx",
                    "event",
                    "from_team_id",
                    "from_player_id",
                    "to_team_id",
                    "to_player_id",
                ],
            )
            writer.writeheader()
            for e in self._events:
                writer.writerow(
                    {
                        "frame_idx": int(e.frame_idx),
                        "event": e.event,
                        "from_team_id": (int(e.from_team) + 1) if e.from_team is not None else "",
                        "from_player_id": int(e.from_player) if e.from_player is not None else "",
                        "to_team_id": (int(e.to_team) + 1) if e.to_team is not None else "",
                        "to_player_id": int(e.to_player) if e.to_player is not None else "",
                    }
                )

        # Team summary
        team_ids = sorted(set(self._team_possession_frames) | set(self._team_passes) | set(self._team_intercepts))
        total_pos_frames = sum(self._team_possession_frames.values())
        with team_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "team_id",
                    "passes",
                    "intercepts",
                    "possession_frames",
                    "possession_seconds",
                    "possession_pct",
                ],
            )
            writer.writeheader()
            for team_id in team_ids:
                pos_frames = int(self._team_possession_frames.get(team_id, 0))
                pos_seconds = float(pos_frames) / max(self.fps, 1e-6)
                pct = (float(pos_frames) / float(total_pos_frames)) if total_pos_frames > 0 else 0.0
                writer.writerow(
                    {
                        "team_id": int(team_id) + 1,
                        "passes": int(self._team_passes.get(team_id, 0)),
                        "intercepts": int(self._team_intercepts.get(team_id, 0)),
                        "possession_frames": pos_frames,
                        "possession_seconds": round(pos_seconds, 3),
                        "possession_pct": round(pct * 100.0, 3),
                    }
                )

        # Pass frequency over time (binned)
        bin_s = max(self.pass_freq_bin_seconds, 0.1)
        counts: dict[tuple[int, int], int] = {}
        max_bin = 0
        for team_id, fidx in self._pass_events:
            t = float(fidx) / max(self.fps, 1e-6)
            b = int(t // bin_s)
            counts[(int(team_id), b)] = counts.get((int(team_id), b), 0) + 1
            max_bin = max(max_bin, b)

        with freq_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "team_id",
                    "bin_start_seconds",
                    "bin_end_seconds",
                    "passes",
                    "passes_per_min",
                ],
            )
            writer.writeheader()
            for team_id in team_ids if team_ids else [0, 1]:
                for b in range(0, max_bin + 1):
                    p = int(counts.get((int(team_id), b), 0))
                    start_s = float(b) * bin_s
                    end_s = start_s + bin_s
                    writer.writerow(
                        {
                            "team_id": int(team_id) + 1,
                            "bin_start_seconds": round(start_s, 3),
                            "bin_end_seconds": round(end_s, 3),
                            "passes": p,
                            "passes_per_min": round((p / bin_s) * 60.0, 3),
                        }
                    )

        return summary_path, events_path, team_path, freq_path

    def get_team_snapshot(self) -> list[dict[str, float | int]]:
        """Return cumulative team stats up to the current frame.

        Intended for real-time overlays while rendering the output video.

        Returns a list of dicts with keys:
          - team_id (1-based)
          - passes
          - possession_pct (0-100)
        """

        # Prefer teams we've actually seen; default to two teams.
        team_ids = sorted(
            set(self._team_possession_frames)
            | set(self._team_passes)
            | set(self._team_intercepts)
        )
        if not team_ids:
            team_ids = [0, 1]

        total_pos_frames = int(sum(self._team_possession_frames.values()))
        snapshot: list[dict[str, float | int]] = []
        for team_id in team_ids:
            pos_frames = int(self._team_possession_frames.get(team_id, 0))
            pct = (float(pos_frames) / float(total_pos_frames) * 100.0) if total_pos_frames > 0 else 0.0
            snapshot.append(
                {
                    "team_id": int(team_id) + 1,
                    "passes": int(self._team_passes.get(team_id, 0)),
                    "possession_pct": float(pct),
                }
            )
        return snapshot

    def get_current_possession_team_id(self) -> int | None:
        """Return current possessing team (1-based) if known."""

        if self._current_team is None:
            return None
        return int(self._current_team) + 1

    def get_last_activity_text(self) -> str | None:
        """Return a human-readable last activity string for overlays."""

        return self._last_activity_text
