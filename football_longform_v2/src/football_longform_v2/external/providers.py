from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalCommand:
    argv: tuple[str, ...]
    cwd: Path


def _resolve(project_root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


@dataclass(frozen=True)
class YoloStructureProvider:
    project_root: Path
    checkpoint: Path
    sample_fps: float = 2.0
    person_class: int = 0
    goal_class: int = 1
    field_line_class: int = 2
    ball_class: int = -1

    @classmethod
    def from_config(cls, raw: dict, subproject_root: Path) -> "YoloStructureProvider":
        return cls(
            project_root=_resolve(subproject_root, raw["project_root"]),
            checkpoint=_resolve(subproject_root, raw["checkpoint"]),
            sample_fps=float(raw.get("sample_fps", 2.0)),
            person_class=int(raw.get("person_class", 0)),
            goal_class=int(raw.get("goal_class", 1)),
            field_line_class=int(raw.get("field_line_class", 2)),
            ball_class=int(raw.get("ball_class", -1)),
        )

    def validate(self) -> None:
        if not (self.project_root / "run_detection_tracking.py").is_file():
            raise FileNotFoundError(self.project_root / "run_detection_tracking.py")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

    def command(self, source: Path, output_dir: Path, device: str = "0") -> ExternalCommand:
        self.validate()
        argv = (
            "python",
            "run_detection_tracking.py",
            "--source", str(source.resolve()),
            "--out-dir", str(output_dir.resolve()),
            "--weights", str(self.checkpoint),
            "--device", device,
            "--imgsz", "1920",
            "--frame-stride", "1",
            "--sample-fps", str(self.sample_fps),
            "--person-cls", str(self.person_class),
            "--goal-cls", str(self.goal_class),
            "--field-line-cls", str(self.field_line_class),
            "--ball-cls", str(self.ball_class),
        )
        return ExternalCommand(argv=argv, cwd=self.project_root)


@dataclass(frozen=True)
class RfDetrBallTeacher:
    project_root: Path
    checkpoint: Path
    preset: Path

    @classmethod
    def from_config(cls, raw: dict, subproject_root: Path) -> "RfDetrBallTeacher":
        return cls(
            project_root=_resolve(subproject_root, raw["project_root"]),
            checkpoint=_resolve(subproject_root, raw["checkpoint"]),
            preset=_resolve(subproject_root, raw["preset"]),
        )

    @property
    def engine_root(self) -> Path:
        return self.project_root / "engines" / "rf_detr"

    def validate(self) -> None:
        required = (
            self.engine_root / "inference_rf_detr_model.py",
            self.checkpoint,
            self.preset,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing RF-DETR inputs: {missing}")

    def command(self, source: Path, output_dir: Path, device: str = "0") -> ExternalCommand:
        self.validate()
        argv = (
            "uv", "run", "python", "inference_rf_detr_model.py",
            "--config", str(self.preset),
            "--source", str(source.resolve()),
            "--output-dir", str(output_dir.resolve()),
            "--checkpoint", str(self.checkpoint),
            "--device", device,
            "--track",
            "--yes",
        )
        return ExternalCommand(argv=argv, cwd=self.engine_root)
