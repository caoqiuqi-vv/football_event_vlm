"""On-demand ball/goal Teacher targets for sampled football training frames."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


class OnlineObjectTeacherTargeter:
    """Fill missing object heatmap targets directly from the sampled RGB batch.

    The dataloader keeps inputs as uint8 RGB when normalize_on_device is enabled,
    so the detector Teachers can reuse exactly the frames consumed by DINO.
    Existing offline targets remain untouched; only frames whose masks are
    entirely absent are inferred online.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        ball_checkpoint: str,
        goal_checkpoint: str,
        patch_size: int,
        ball_confidence: float,
        goal_confidence: float,
        ball_sigma_patches: float,
        goal_dilation_patches: float,
        input_size: Sequence[int] = (1088, 1920),
        batch_size: int = 16,
        half: bool = True,
        iou: float = 0.70,
    ) -> None:
        self.device = torch.device(device)
        self.ball_checkpoint = str(Path(ball_checkpoint).expanduser())
        self.goal_checkpoint = str(Path(goal_checkpoint).expanduser())
        self.patch_size = int(patch_size)
        self.ball_confidence = float(ball_confidence)
        self.goal_confidence = float(goal_confidence)
        self.ball_sigma_patches = max(float(ball_sigma_patches), 0.25)
        self.goal_dilation_patches = max(float(goal_dilation_patches), 0.0)
        self.input_height, self.input_width = (int(value) for value in input_size)
        if self.input_height % 32 or self.input_width % 32:
            raise ValueError(
                "online object Teacher input_size must be divisible by 32: "
                f"{tuple(input_size)}"
            )
        self.batch_size = max(int(batch_size), 1)
        self.half = bool(half)
        self.iou = float(iou)
        self.ball_model: Any | None = None
        self.goal_model: Any | None = None
        self.ball_class_id = -1
        self.goal_class_id = -1

    def _ensure_models(self) -> None:
        if self.ball_model is not None and self.goal_model is not None:
            return
        from ultralytics import YOLO

        self.ball_model = YOLO(self.ball_checkpoint)
        self.goal_model = YOLO(self.goal_checkpoint)
        ball_ids = [
            int(key)
            for key, value in self.ball_model.names.items()
            if str(value).casefold()
            in {"football", "soccer_ball", "soccer ball", "ball"}
        ]
        goal_ids = [
            int(key)
            for key, value in self.goal_model.names.items()
            if str(value).casefold() == "goal"
        ]
        if len(ball_ids) != 1:
            raise ValueError(
                f"online ball checkpoint must expose one football class: "
                f"{self.ball_model.names}"
            )
        if len(goal_ids) != 1:
            raise ValueError(
                f"online goal checkpoint must expose one goal class: "
                f"{self.goal_model.names}"
            )
        self.ball_class_id = ball_ids[0]
        self.goal_class_id = goal_ids[0]

    def _predict(
        self,
        model: Any,
        frames: Tensor,
        *,
        class_id: int,
        confidence: float,
    ) -> list[Any]:
        return model.predict(
            frames,
            imgsz=(self.input_height, self.input_width),
            conf=float(confidence),
            iou=self.iou,
            classes=[int(class_id)],
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

    def _ball_target(
        self,
        box: Sequence[float],
        confidence: float,
        *,
        grid_h: int,
        grid_w: int,
        grid_y: Tensor,
        grid_x: Tensor,
    ) -> Tensor:
        cx = (float(box[0]) + float(box[2])) * 0.5 / self.input_width * grid_w
        cy = (float(box[1]) + float(box[3])) * 0.5 / self.input_height * grid_h
        distance2 = (grid_x - cx).square() + (grid_y - cy).square()
        return float(confidence) * torch.exp(
            -0.5 * distance2 / (self.ball_sigma_patches**2)
        )

    def _goal_target(
        self,
        box: Sequence[float],
        confidence: float,
        *,
        grid_h: int,
        grid_w: int,
        grid_y: Tensor,
        grid_x: Tensor,
    ) -> Tensor:
        x1 = float(box[0]) / self.input_width * grid_w - self.goal_dilation_patches
        y1 = float(box[1]) / self.input_height * grid_h - self.goal_dilation_patches
        x2 = float(box[2]) / self.input_width * grid_w + self.goal_dilation_patches
        y2 = float(box[3]) / self.input_height * grid_h + self.goal_dilation_patches
        zero = torch.zeros_like(grid_x)
        dx = torch.maximum(torch.maximum(x1 - grid_x, grid_x - x2), zero)
        dy = torch.maximum(torch.maximum(y1 - grid_y, grid_y - y2), zero)
        distance = torch.sqrt(dx.square() + dy.square())
        return float(confidence) * (1.0 - distance).clamp(0.0, 1.0)

    @torch.inference_mode()
    def fill_missing(self, batch: dict[str, Any]) -> dict[str, float]:
        targets = batch.get("object_heatmap_targets")
        masks = batch.get("object_heatmap_masks")
        inputs = batch.get("inputs")
        if not torch.is_tensor(targets) or not torch.is_tensor(masks):
            raise ValueError("online object Teacher requires heatmap targets and masks")
        if not torch.is_tensor(inputs) or inputs.ndim != 5:
            raise ValueError("online object Teacher requires inputs shaped [B,T,C,H,W]")
        if targets.shape != masks.shape or targets.ndim != 4 or targets.shape[-1] != 2:
            raise ValueError(
                f"invalid object target shape targets={tuple(targets.shape)} "
                f"masks={tuple(masks.shape)}"
            )
        batch_size, frames_per_clip, patch_count, _ = targets.shape
        missing = masks.amax(dim=(2, 3)) < 0.5
        flat_missing = torch.nonzero(missing.flatten(), as_tuple=False).flatten()
        if not len(flat_missing):
            return {
                "online_object_teacher_frames": 0.0,
                "online_object_teacher_fraction": 0.0,
                "online_object_teacher_seconds": 0.0,
            }

        self._ensure_models()
        grid_h = int(round((patch_count * inputs.shape[-2] / inputs.shape[-1]) ** 0.5))
        if grid_h <= 0 or patch_count % grid_h:
            raise ValueError(f"cannot infer patch grid from patch_count={patch_count}")
        grid_w = patch_count // grid_h
        if grid_h * self.patch_size != inputs.shape[-2] or grid_w * self.patch_size != inputs.shape[-1]:
            raise ValueError(
                f"patch grid mismatch input={tuple(inputs.shape[-2:])} "
                f"grid={(grid_h, grid_w)} patch={self.patch_size}"
            )
        grid_y, grid_x = self._grid(grid_h, grid_w)
        flat_inputs = inputs.reshape(
            batch_size * frames_per_clip, *inputs.shape[2:]
        )
        started = time.perf_counter()
        for offset in range(0, len(flat_missing), self.batch_size):
            selected_indices = flat_missing[offset : offset + self.batch_size]
            selected = flat_inputs.index_select(0, selected_indices).to(
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
            ball_results = self._predict(
                self.ball_model,
                selected,
                class_id=self.ball_class_id,
                confidence=self.ball_confidence,
            )
            goal_results = self._predict(
                self.goal_model,
                selected,
                class_id=self.goal_class_id,
                confidence=self.goal_confidence,
            )
            for local_index, flat_index_tensor in enumerate(selected_indices):
                flat_index = int(flat_index_tensor)
                sample_index = flat_index // frames_per_clip
                frame_index = flat_index % frames_per_clip
                frame_target = torch.zeros(
                    (grid_h, grid_w, 2), dtype=torch.float32
                )
                for object_index, results in enumerate((ball_results, goal_results)):
                    boxes = results[local_index].boxes
                    if boxes is None:
                        continue
                    for score, box in zip(
                        boxes.conf.detach().cpu().tolist(),
                        boxes.xyxy.detach().cpu().tolist(),
                    ):
                        if object_index == 0:
                            contribution = self._ball_target(
                                box,
                                float(score),
                                grid_h=grid_h,
                                grid_w=grid_w,
                                grid_y=grid_y,
                                grid_x=grid_x,
                            )
                        else:
                            contribution = self._goal_target(
                                box,
                                float(score),
                                grid_h=grid_h,
                                grid_w=grid_w,
                                grid_y=grid_y,
                                grid_x=grid_x,
                            )
                        frame_target[..., object_index] = torch.maximum(
                            frame_target[..., object_index], contribution
                        )
                targets[sample_index, frame_index] = frame_target.reshape(
                    patch_count, 2
                ).to(targets.dtype)
                masks[sample_index, frame_index] = 1.0

        elapsed = time.perf_counter() - started
        batch["object_heatmap_targets"] = targets
        batch["object_heatmap_masks"] = masks
        inferred = float(len(flat_missing))
        return {
            "online_object_teacher_frames": inferred,
            "online_object_teacher_fraction": inferred
            / float(batch_size * frames_per_clip),
            "online_object_teacher_seconds": float(elapsed),
        }

