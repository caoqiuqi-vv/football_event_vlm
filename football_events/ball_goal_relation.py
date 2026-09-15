"""Training-only ball/goal relation targets and representation losses.

The event classifier never consumes detector coordinates.  Offline ball and
goal teachers only create masked auxiliary targets.  Relation predictions are
made from the same frame tokens consumed by the temporal event head, which is
the intended transfer route into event understanding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from football_object_spatial_aux import ObjectTeacherTargetProvider
from football_events.tracked_ball_teacher import TrackedBallTargetProvider


RELATION_NAMES = (
    "goal_visible",
    "ball_left_of_goal",
    "ball_inside_goal_x",
    "ball_right_of_goal",
    "ball_above_goal",
    "ball_inside_goal_y",
    "ball_below_goal",
    "ball_inside_goal",
    "ball_near_goal",
    "ball_far_from_goal",
)


class BallGoalRelationTargetProvider:
    """Combine a repaired ball track with an independently supervised goal.

    Ball localization keeps the v9 quality-weighted target.  Goal localization
    comes from the high-coverage compact ROI index.  Relation targets are
    enabled only for high-confidence, high-quality ball rows with a visible
    goal; ambiguous or unavailable frames remain unknown instead of negative.
    """

    def __init__(
        self,
        ball_index_root: str | Path,
        goal_index_root: str | Path,
        *,
        image_size: Sequence[int],
        patch_size: int = 16,
        max_ball_frame_gap_sec: float = 0.10,
        max_goal_frame_gap_sec: float = 0.12,
        ball_sigma_patches: float = 1.25,
        goal_dilation_patches: float = 0.5,
        ball_min_confidence: float = 0.0,
        ball_min_quality: float = 0.0,
        relation_min_ball_confidence: float = 0.20,
        relation_min_ball_quality: float = 0.75,
        goal_confidence: float = 0.25,
        goal_class_id: int = 2,
        goal_negative_weight: float = 0.05,
        teacher_time_offset_sec: float = 0.0,
        max_cached_videos: int = 4,
    ) -> None:
        self.ball = TrackedBallTargetProvider(
            ball_index_root,
            image_size=image_size,
            patch_size=patch_size,
            max_frame_gap_sec=max_ball_frame_gap_sec,
            ball_sigma_patches=ball_sigma_patches,
            min_confidence=ball_min_confidence,
            min_quality=ball_min_quality,
            teacher_time_offset_sec=teacher_time_offset_sec,
            max_cached_videos=max_cached_videos,
        )
        self.goal = ObjectTeacherTargetProvider(
            goal_index_root,
            image_size=image_size,
            patch_size=patch_size,
            # The ball channel is discarded.  Keep its threshold impossible so
            # accidental class-map changes cannot leak into the composite.
            ball_class_id=-999,
            goal_class_id=goal_class_id,
            ball_confidence=2.0,
            goal_confidence=goal_confidence,
            max_frame_gap_sec=max_goal_frame_gap_sec,
            ball_sigma_patches=ball_sigma_patches,
            goal_dilation_patches=goal_dilation_patches,
            max_cached_videos=max_cached_videos,
        )
        self.relation_min_ball_confidence = max(
            float(relation_min_ball_confidence), 0.0
        )
        self.relation_min_ball_quality = max(
            float(relation_min_ball_quality), 0.0
        )
        self.goal_negative_weight = max(float(goal_negative_weight), 0.0)
        self.patch_count = self.ball.patch_count
        self.grid_h = self.ball.grid_h
        self.grid_w = self.ball.grid_w

    def has_video(self, video_id: str) -> bool:
        return self.ball.has_video(video_id) or self.goal.has_video(video_id)

    def empty(self, frames: int) -> tuple[Tensor, Tensor]:
        return self.ball.empty(frames)

    def empty_bundle(self, frames: int) -> dict[str, Tensor]:
        object_targets, object_masks = self.empty(frames)
        return {
            "object_heatmap_targets": object_targets,
            "object_heatmap_masks": object_masks,
            "ball_goal_relation_targets": torch.zeros(
                int(frames), len(RELATION_NAMES), dtype=torch.float32
            ),
            "ball_goal_relation_masks": torch.zeros(
                int(frames), len(RELATION_NAMES), dtype=torch.float32
            ),
            "ball_context_masks": torch.zeros(
                int(frames), dtype=torch.float32
            ),
        }

    def targets(
        self, video_id: str, absolute_times: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        bundle = self.target_bundle(video_id, absolute_times)
        return bundle["object_heatmap_targets"], bundle["object_heatmap_masks"]

    def target_bundle(
        self, video_id: str, absolute_times: Sequence[float]
    ) -> dict[str, Tensor]:
        ball_targets, ball_masks, ball_metadata = self.ball.targets_with_metadata(
            video_id, absolute_times
        )
        goal_targets, goal_masks = self.goal.targets(video_id, absolute_times)
        targets = ball_targets.clone()
        masks = ball_masks.clone()
        targets[..., 1] = goal_targets[..., 1]
        masks[..., 1] = goal_masks[..., 1]

        frames = len(absolute_times)
        relation_targets = torch.zeros(
            frames, len(RELATION_NAMES), dtype=torch.float32
        )
        relation_masks = torch.zeros_like(relation_targets)
        context_masks = torch.zeros(frames, dtype=torch.float32)
        goal_channel = targets[..., 1].reshape(frames, self.grid_h, self.grid_w)
        goal_valid = masks[..., 1].amax(dim=1) > 0
        goal_strength = goal_channel.amax(dim=(1, 2))
        goal_present = goal_strength >= 0.05

        # Goal visibility includes weak negatives only on frames known to have
        # been processed by the goal teacher.  Relation geometry never treats
        # a detector miss as a negative label.
        relation_targets[:, 0] = goal_present.float()
        relation_masks[:, 0] = torch.where(
            goal_present,
            goal_strength.clamp(0.0, 1.0),
            torch.full_like(goal_strength, self.goal_negative_weight),
        ) * goal_valid.float()

        ball_boxes = ball_metadata["bbox_xyxy_norm"]
        ball_confidence = ball_metadata["confidence"]
        ball_quality = ball_metadata["quality"]
        ball_valid = ball_metadata["valid"] > 0
        pair_valid = (
            ball_valid
            & (ball_confidence >= self.relation_min_ball_confidence)
            & (ball_quality >= self.relation_min_ball_quality)
            & goal_present
            & goal_valid
        )
        for frame_index in torch.nonzero(pair_valid, as_tuple=False).flatten().tolist():
            heatmap = goal_channel[frame_index]
            positive = heatmap >= 0.05
            ys, xs = torch.nonzero(positive, as_tuple=True)
            if not len(xs):
                continue
            x1 = float(xs.min()) / self.grid_w
            x2 = float(xs.max() + 1) / self.grid_w
            y1 = float(ys.min()) / self.grid_h
            y2 = float(ys.max() + 1) / self.grid_h
            bx1, by1, bx2, by2 = (
                float(value) for value in ball_boxes[frame_index]
            )
            ball_x = 0.5 * (bx1 + bx2)
            ball_y = 0.5 * (by1 + by2)
            x_class = 0 if ball_x < x1 else 2 if ball_x > x2 else 1
            y_class = 0 if ball_y < y1 else 2 if ball_y > y2 else 1
            relation_targets[frame_index, 1 + x_class] = 1.0
            relation_targets[frame_index, 4 + y_class] = 1.0

            dx = max(x1 - ball_x, 0.0, ball_x - x2)
            dy = max(y1 - ball_y, 0.0, ball_y - y2)
            goal_scale = max(x2 - x1, y2 - y1, 1.0 / max(self.grid_h, self.grid_w))
            normalized_distance = (dx * dx + dy * dy) ** 0.5 / goal_scale
            distance_class = 0 if normalized_distance == 0.0 else 1 if normalized_distance <= 2.0 else 2
            relation_targets[frame_index, 7 + distance_class] = 1.0
            pair_weight = min(
                float(ball_quality[frame_index]),
                float(goal_strength[frame_index]),
            )
            relation_masks[frame_index, 1:] = pair_weight
            context_masks[frame_index] = pair_weight

        return {
            "object_heatmap_targets": targets,
            "object_heatmap_masks": masks,
            "ball_goal_relation_targets": relation_targets,
            "ball_goal_relation_masks": relation_masks,
            "ball_context_masks": context_masks,
        }


def ball_goal_relation_aux_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Supervise relation semantics on event frame tokens.

    The context target is a detached DINO patch embedding pooled around the
    model's Top-K ball hypotheses.  Predicting it from event frame tokens makes
    those tokens retain local player/contact context without player boxes.
    """

    logits = outputs.get("ball_goal_relation_logits")
    context_predictions = outputs.get("ball_context_predictions")
    context_targets = outputs.get("ball_context_targets")
    if logits is None or context_predictions is None or context_targets is None:
        raise ValueError(
            "ball-goal relation loss requires relation logits and context tensors"
        )
    targets = batch["ball_goal_relation_targets"].to(
        device, non_blocking=True
    ).to(logits.dtype)
    masks = batch["ball_goal_relation_masks"].to(
        device, non_blocking=True
    ).to(logits.dtype)
    if logits.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError(
            "ball-goal relation shape mismatch: "
            f"logits={tuple(logits.shape)} targets={tuple(targets.shape)} "
            f"masks={tuple(masks.shape)}"
        )
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    relation_loss = (bce * masks).sum() / masks.sum().clamp_min(1.0)

    context_masks = batch["ball_context_masks"].to(
        device, non_blocking=True
    ).to(context_predictions.dtype)
    if context_predictions.shape != context_targets.shape:
        raise ValueError(
            "ball context shape mismatch: "
            f"pred={tuple(context_predictions.shape)} "
            f"target={tuple(context_targets.shape)}"
        )
    context_distance = 1.0 - F.cosine_similarity(
        context_predictions.float(), context_targets.detach().float(), dim=-1
    )
    expanded_context_masks = context_masks.unsqueeze(-1).expand_as(context_distance)
    context_loss = (context_distance * expanded_context_masks).sum() / (
        expanded_context_masks.sum().clamp_min(1.0)
    )
    model_cfg = cfg.get("model", {})
    relation_cfg = model_cfg.get("ball_goal_relation_aux", {})
    context_weight = float(relation_cfg.get("context_loss_weight", 0.25))
    loss = relation_loss + context_weight * context_loss

    with torch.no_grad():
        predictions = logits.sigmoid() >= 0.5
        labels = targets >= 0.5
        active = masks > 0
        accuracy = ((predictions == labels) & active).sum() / active.sum().clamp_min(1)
        pair_frames = (masks[..., 1:].amax(dim=-1) > 0).float().mean()
        goal_frames = (masks[..., 0] > 0).float().mean()
    return loss, {
        "ball_goal_relation_loss": float(loss.detach().cpu()),
        "ball_goal_relation_bce": float(relation_loss.detach().cpu()),
        "ball_context_loss": float(context_loss.detach().cpu()),
        "ball_goal_relation_accuracy": float(accuracy.detach().cpu()),
        "ball_goal_pair_frame_fraction": float(pair_frames.detach().cpu()),
        "goal_teacher_frame_fraction": float(goal_frames.detach().cpu()),
        "ball_candidate_entropy": float(
            outputs["ball_candidate_entropy"].float().mean().detach().cpu()
        ),
    }
