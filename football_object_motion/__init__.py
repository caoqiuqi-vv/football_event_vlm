"""Object-motion evidence branch for football event recognition."""

from .losses import object_motion_auxiliary_loss
from .model import OBJECT_NAMES, ObjectMotionEvidenceAdapter
from .teacher import OnlineObjectMotionTeacher

__all__ = [
    "OBJECT_NAMES",
    "ObjectMotionEvidenceAdapter",
    "OnlineObjectMotionTeacher",
    "object_motion_auxiliary_loss",
]
