"""Online ball/goal/person Teacher for dense object-motion frames."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import OBJECT_NAMES


def ball_teacher_quality(confidences: Tensor, centers: Tensor, *, jump_limit: float = 0.35) -> dict[str, Tensor]:
    """Tri-state ball supervision with conservative trajectory upgrades."""
    if confidences.ndim != 2 or centers.shape != (*confidences.shape, 2):
        raise ValueError("ball quality expects confidences [B,T], centers [B,T,2]")
    strong = confidences >= 0.30
    weak = (confidences >= 0.10) & ~strong
    observed = confidences >= 0.10
    step_ok = torch.zeros_like(observed)
    if confidences.shape[1] > 1:
        jumps = (centers[:, 1:] - centers[:, :-1]).norm(dim=-1)
        step_ok[:, 1:] = observed[:, 1:] & observed[:, :-1] & (jumps <= jump_limit)
    consistent = torch.zeros_like(observed)
    if confidences.shape[1] > 2:
        triplet = step_ok[:, 1:-1] & step_ok[:, 2:]
        consistent[:, 1:-1] |= triplet
        consistent[:, :-2] |= triplet
        consistent[:, 2:] |= triplet
    upgraded = weak & consistent
    positive_quality = torch.where(strong | upgraded, torch.ones_like(confidences), torch.where(weak, torch.full_like(confidences, 0.25), torch.zeros_like(confidences)))
    presence_quality = torch.where(observed, positive_quality, torch.full_like(confidences, 0.05))
    neighbor_ok = step_ok.clone()
    if confidences.shape[1] > 1:
        neighbor_ok[:, :-1] |= step_ok[:, 1:]
    coordinate_quality = ((strong & neighbor_ok) | upgraded).to(confidences.dtype)
    motion_quality = coordinate_quality * step_ok.to(confidences.dtype)
    return {"positive": positive_quality, "presence": presence_quality, "coordinate": coordinate_quality, "motion": motion_quality, "upgraded": upgraded.to(confidences.dtype)}


class OnlineObjectMotionTeacher:
    """Generate patch heatmaps, presence and coordinates from YOLO Teachers."""

    def __init__(
        self,
        *,
        device: torch.device,
        ball_checkpoint: str,
        scene_checkpoint: str,
        patch_size: int,
        ball_confidence: float = 0.10,
        goal_confidence: float = 0.25,
        person_confidence: float = 0.25,
        ball_sigma_patches: float = 1.25,
        box_dilation_patches: float = 0.5,
        input_size: Sequence[int] = (1088, 1920),
        batch_size: int = 16,
        half: bool = True,
        iou: float = 0.70,
        absence_presence_weights: Sequence[float] = (0.05, 0.02, 0.02),
        enabled_objects: Sequence[str] = OBJECT_NAMES,
    ) -> None:
        self.device = torch.device(device)
        self.ball_checkpoint = str(Path(ball_checkpoint).expanduser())
        self.scene_checkpoint = str(Path(scene_checkpoint).expanduser())
        self.patch_size = int(patch_size)
        self.confidences = {
            "ball": float(ball_confidence),
            "goal": float(goal_confidence),
            "person": float(person_confidence),
        }
        self.ball_sigma_patches = max(float(ball_sigma_patches), 0.25)
        self.box_dilation_patches = max(float(box_dilation_patches), 0.0)
        self.input_height, self.input_width = (int(value) for value in input_size)
        self.batch_size = max(int(batch_size), 1)
        self.half = bool(half)
        self.iou = float(iou)
        if len(absence_presence_weights) != len(OBJECT_NAMES):
            raise ValueError(
                "absence_presence_weights must match ball/goal/person"
            )
        self.absence_presence_weights = tuple(
            max(float(value), 0.0) for value in absence_presence_weights
        )
        self.enabled_objects = frozenset(str(value) for value in enabled_objects)
        unknown = self.enabled_objects.difference(OBJECT_NAMES)
        if unknown or not self.enabled_objects:
            raise ValueError(
                f"enabled_objects must be a non-empty subset of {OBJECT_NAMES}; got {sorted(self.enabled_objects)}"
            )
        self.ball_model: Any | None = None
        self.scene_model: Any | None = None
        self.class_ids: dict[str, int] = {}

        self._models_ready = False
    @staticmethod
    def empty(
        batch_size: int,
        frames: int,
        patch_count: int,
    ) -> dict[str, Tensor]:
        return {
            "object_motion_heatmap_targets": torch.zeros(
                batch_size, frames, patch_count, len(OBJECT_NAMES), dtype=torch.float32
            ),
            "object_motion_heatmap_masks": torch.zeros(
                batch_size, frames, patch_count, len(OBJECT_NAMES), dtype=torch.float32
            ),
            "object_motion_presence_targets": torch.zeros(
                batch_size, frames, len(OBJECT_NAMES), dtype=torch.float32
            ),
            "object_motion_presence_masks": torch.zeros(
                batch_size, frames, len(OBJECT_NAMES), dtype=torch.float32
            ),
            "object_motion_coordinate_targets": torch.zeros(
                batch_size, frames, len(OBJECT_NAMES), 4, dtype=torch.float32
            ),
            "object_motion_coordinate_masks": torch.zeros(
                batch_size, frames, len(OBJECT_NAMES), dtype=torch.float32
            ),
            "object_motion_teacher_confidences": torch.zeros(
                batch_size, frames, len(OBJECT_NAMES), dtype=torch.float32
            ),
            "object_motion_motion_quality": torch.zeros(
                batch_size, frames, len(OBJECT_NAMES), dtype=torch.float32
            ),
        }

    @staticmethod
    def _find_class_id(names: dict[int, str], accepted: set[str], label: str) -> int:
        matches = [
            int(index)
            for index, name in names.items()
            if str(name).casefold().strip() in accepted
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Teacher must expose exactly one {label} class; names={names}"
            )
        return matches[0]

    def _ensure_models(self) -> None:
        if self._models_ready:
            return
        from ultralytics import YOLO

        if "ball" in self.enabled_objects:
            self.ball_model = YOLO(self.ball_checkpoint)
            self.class_ids["ball"] = self._find_class_id(
                self.ball_model.names,
                {"football", "soccer_ball", "soccer ball", "ball"},
                "ball",
            )
        scene_objects = self.enabled_objects.intersection({"goal", "person"})
        if scene_objects:
            self.scene_model = YOLO(self.scene_checkpoint)
            if "goal" in scene_objects:
                self.class_ids["goal"] = self._find_class_id(
                    self.scene_model.names, {"goal", "goalpost", "goal post"}, "goal"
                )
            if "person" in scene_objects:
                self.class_ids["person"] = self._find_class_id(
                    self.scene_model.names,
                    {"person", "player", "football player", "soccer player"},
                    "person",
                )
        self._models_ready = True

    def _predict(
        self,
        model: Any,
        frames: Tensor,
        *,
        classes: Sequence[int],
        confidence: float,
    ) -> list[Any]:
        return model.predict(
            frames,
            imgsz=(self.input_height, self.input_width),
            conf=float(confidence),
            iou=self.iou,
            classes=[int(value) for value in classes],
            device=str(self.device),
            half=self.half,
            verbose=False,
        )

    @staticmethod
    def _grid(grid_h: int, grid_w: int) -> tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32) + 0.5,
            torch.arange(grid_w, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        return yy, xx

    def _target_from_box(
        self,
        object_name: str,
        box: Sequence[float],
        confidence: float,
        *,
        grid_h: int,
        grid_w: int,
        grid_y: Tensor,
        grid_x: Tensor,
    ) -> Tensor:
        if object_name == "ball":
            cx = (float(box[0]) + float(box[2])) * 0.5 / self.input_width * grid_w
            cy = (float(box[1]) + float(box[3])) * 0.5 / self.input_height * grid_h
            distance2 = (grid_x - cx).square() + (grid_y - cy).square()
            return torch.exp(
                -0.5 * distance2 / (self.ball_sigma_patches**2)
            )
        x1 = float(box[0]) / self.input_width * grid_w - self.box_dilation_patches
        y1 = float(box[1]) / self.input_height * grid_h - self.box_dilation_patches
        x2 = float(box[2]) / self.input_width * grid_w + self.box_dilation_patches
        y2 = float(box[3]) / self.input_height * grid_h + self.box_dilation_patches
        zero = torch.zeros_like(grid_x)
        dx = torch.maximum(torch.maximum(x1 - grid_x, grid_x - x2), zero)
        dy = torch.maximum(torch.maximum(y1 - grid_y, grid_y - y2), zero)
        distance = torch.sqrt(dx.square() + dy.square())
        return (1.0 - distance).clamp(0.0, 1.0)

    def fill(self, batch: dict[str, Any]) -> dict[str, float]:
        inputs = batch.get("object_motion_inputs")
        if not torch.is_tensor(inputs) or inputs.ndim != 5:
            raise ValueError("object motion Teacher requires inputs [B,T,C,H,W]")
        targets = batch.get("object_motion_heatmap_targets")
        masks = batch.get("object_motion_heatmap_masks")
        if not torch.is_tensor(targets) or not torch.is_tensor(masks):
            raise ValueError("object motion Teacher target placeholders are missing")
        batch_size, frames, patch_count, object_count = targets.shape
        if object_count != len(OBJECT_NAMES):
            raise ValueError("object motion Teacher object channel mismatch")
        self._ensure_models()
        grid_h = int(inputs.shape[-2]) // self.patch_size
        grid_w = int(inputs.shape[-1]) // self.patch_size
        if grid_h * grid_w != patch_count:
            raise ValueError(
                f"object motion patch grid mismatch input={tuple(inputs.shape[-2:])} "
                f"patches={patch_count}"
            )
        teacher_inputs = batch.get("object_motion_teacher_inputs", inputs)
        if not torch.is_tensor(teacher_inputs) or teacher_inputs.shape[:2] != inputs.shape[:2]:
            raise ValueError("object motion high-resolution Teacher inputs must match [B,T]")
        flat = teacher_inputs.reshape(batch_size * frames, *teacher_inputs.shape[2:])
        grid_y, grid_x = self._grid(grid_h, grid_w)
        presence_targets = batch["object_motion_presence_targets"]
        presence_masks = batch["object_motion_presence_masks"]
        coordinate_targets = batch["object_motion_coordinate_targets"]
        coordinate_masks = batch["object_motion_coordinate_masks"]
        teacher_confidences = batch["object_motion_teacher_confidences"]
        started = time.perf_counter()
        for offset in range(0, flat.shape[0], self.batch_size):
            selected = flat[offset : offset + self.batch_size].to(
                self.device, non_blocking=True
            )
            if selected.dtype == torch.uint8:
                selected = selected.float().div_(255.0)
            else:
                selected = selected.float()
            selected = F.interpolate(
                selected,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
            ball_results = None
            if "ball" in self.enabled_objects:
                ball_results = self._predict(
                    self.ball_model,
                    selected,
                    classes=[self.class_ids["ball"]],
                    confidence=self.confidences["ball"],
                )
            scene_objects = [
                name for name in ("goal", "person") if name in self.enabled_objects
            ]
            scene_results = None
            if scene_objects:
                scene_results = self._predict(
                    self.scene_model,
                    selected,
                    classes=[self.class_ids[name] for name in scene_objects],
                    confidence=min(self.confidences[name] for name in scene_objects),
                )
            for local_index in range(selected.shape[0]):
                flat_index = offset + local_index
                sample_index = flat_index // frames
                frame_index = flat_index % frames
                by_name = {
                    "ball": ball_results[local_index].boxes if ball_results is not None else None,
                    "goal": scene_results[local_index].boxes if scene_results is not None else None,
                    "person": scene_results[local_index].boxes if scene_results is not None else None,
                }
                for object_index, object_name in enumerate(OBJECT_NAMES):
                    if object_name not in self.enabled_objects:
                        continue
                    boxes = by_name[object_name]
                    candidates: list[tuple[float, Sequence[float]]] = []
                    if boxes is not None:
                        class_values = boxes.cls.detach().cpu().tolist()
                        for class_value, score, box in zip(
                            class_values,
                            boxes.conf.detach().cpu().tolist(),
                            boxes.xyxy.detach().cpu().tolist(),
                        ):
                            if int(class_value) != self.class_ids[object_name]:
                                continue
                            if float(score) < self.confidences[object_name]:
                                continue
                            candidates.append((float(score), box))
                    heatmap = torch.zeros(grid_h, grid_w, dtype=torch.float32)
                    for score, box in candidates:
                        heatmap = torch.maximum(
                            heatmap,
                            self._target_from_box(
                                object_name,
                                box,
                                score,
                                grid_h=grid_h,
                                grid_w=grid_w,
                                grid_y=grid_y,
                                grid_x=grid_x,
                            ),
                        )
                    targets[sample_index, frame_index, :, object_index] = heatmap.flatten()
                    if not candidates:
                        continue
                    best_score, best_box = max(candidates, key=lambda item: item[0])
                    presence_targets[sample_index, frame_index, object_index] = 1.0
                    teacher_confidences[sample_index, frame_index, object_index] = best_score
                    masks[sample_index, frame_index, :, object_index] = float(best_score)
                    presence_masks[sample_index, frame_index, object_index] = float(
                        best_score
                    )
                    # A union person heatmap is supervised, but one coordinate
                    # would be ambiguous.  Coordinates are used for ball/goal.
                    if object_name == "person":
                        continue
                    x1, y1, x2, y2 = (float(value) for value in best_box)
                    coordinate_targets[sample_index, frame_index, object_index] = torch.tensor(
                        [
                            ((x1 + x2) / self.input_width) - 1.0,
                            ((y1 + y2) / self.input_height) - 1.0,
                            (x2 - x1) / self.input_width * 2.0,
                            (y2 - y1) / self.input_height * 2.0,
                        ],
                        dtype=torch.float32,
                    )
                    coordinate_masks[sample_index, frame_index, object_index] = float(
                        best_score
                    )
        if "ball" in self.enabled_objects:
            ball_centers = coordinate_targets[:, :, 0, :2]
            quality = ball_teacher_quality(teacher_confidences[:, :, 0], ball_centers)
            masks[:, :, :, 0] = quality["positive"].unsqueeze(-1).expand_as(
                masks[:, :, :, 0]
            )
            presence_masks[:, :, 0] = torch.where(
                teacher_confidences[:, :, 0] >= self.confidences["ball"],
                quality["presence"],
                torch.full_like(
                    quality["presence"], self.absence_presence_weights[0]
                ),
            )
            coordinate_masks[:, :, 0] *= quality["coordinate"]
            batch["object_motion_motion_quality"][:, :, 0] = quality["motion"]
        # Detector misses are unknown for localization, not pseudo-background.
        # Presence keeps a tiny absent weight so no-object tokens can learn.
        for object_index in (1, 2):
            confidence = teacher_confidences[:, :, object_index]
            if OBJECT_NAMES[object_index] not in self.enabled_objects:
                continue
            quality_weight = torch.where(
                confidence > 0, confidence.clamp_min(0.25), torch.zeros_like(confidence)
            )
            masks[:, :, :, object_index] = quality_weight.unsqueeze(-1).expand_as(
                masks[:, :, :, object_index]
            )
            presence_masks[:, :, object_index] = torch.where(
                confidence > 0,
                quality_weight,
                torch.full_like(
                    confidence,
                    self.absence_presence_weights[object_index],
                ),
            )
            if object_index == 1:
                coordinate_masks[:, :, object_index] *= quality_weight
                batch["object_motion_motion_quality"][:, :, object_index] = quality_weight
        elapsed = time.perf_counter() - started
        batch["object_motion_teacher_filled"] = torch.ones(batch_size, frames, dtype=torch.float32)
        batch["object_motion_heatmap_targets"] = targets
        batch["object_motion_heatmap_masks"] = masks
        return {
            "online_object_motion_teacher_frames": float(batch_size * frames),
            "online_object_motion_teacher_seconds": float(elapsed),
            "online_object_motion_teacher_fraction": 1.0,
        }
