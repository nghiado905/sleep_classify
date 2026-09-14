"""Roboflow sources and class mapping used by the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoboflowSource:
    workspace: str
    project: str
    version: int

    @property
    def slug(self) -> str:
        return f"{self.workspace}/{self.project}/{self.version}"


ROBOFLOW_SOURCES = (
    RoboflowSource("ntthinh", "student-behaviour-detection-neazg-fbhsp", 1),
    RoboflowSource("aus-model-2", "student-behaviour-detection-neazg-r6kny", 1),
    RoboflowSource("mywork-lkwz4", "student-behaviour-detection-neazg", 1),
    RoboflowSource("studentclassroombehavior", "sleep-jgims", 1),
    RoboflowSource("classroomviolations", "detection-sleep", 1),
    RoboflowSource("demo-kyv3w", "sleeping-person-in-classroom-exmz9", 2),
    RoboflowSource(
        "student-attention-monitoring-system",
        "student-attention-monitoring",
        1,
    ),
    RoboflowSource("suhas-yc", "classroom-behavior-detection-tfzpo", 1),
)

SLEEP_CLASS_ALIASES = {
    "sleep",
    "sleeping",
    "sleepy",
    "bow_head",
    "bow-head",
    "bowing the head",
    "leaning",
    "leaning over the table",
    "leaning on the desk",
    "drowsy",
    "fatigue",
    "head_drop",
    "lying",
    "sleepingstudent",
}

ROBOFLOW_FORMAT = "yolov8"
DEFAULT_TARGET_PER_CLASS = 8_500
DEFAULT_SEED = 42
