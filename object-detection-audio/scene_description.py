"""Turn DETR boxes into conservative, human-readable scene descriptions."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _position(center_x: float) -> str:
    if center_x < 0.36:
        return "left"
    if center_x > 0.64:
        return "right"
    return "center"


def _count_phrase(label: str, count: int) -> str:
    irregular = {
        "person": "people",
        "sheep": "sheep",
        "mouse": "mice",
        "knife": "knives",
        "scissors": "scissors",
    }
    plural = irregular.get(label, f"{label}s")
    return f"{count} {label if count == 1 else plural}"


def describe_scene(
    detections: list[dict[str, Any]],
    image_size: tuple[int, int],
    return_rows: bool = False,
):
    """Describe only detected COCO labels, counts, confidence, and box positions."""
    width, _ = image_size
    rows = []
    for detection in detections:
        box = detection["box"]
        center_x = (float(box["xmin"]) + float(box["xmax"])) / 2
        location = _position(center_x / max(width, 1))
        rows.append({
            "label": str(detection["label"]),
            "score": float(detection["score"]),
            "location": location,
            "center_x": center_x,
        })

    if not rows:
        text = "I could not identify any objects above the selected confidence threshold."
        return (text, []) if return_rows else text

    counts = Counter(row["label"] for row in rows)
    count_text = ", ".join(_count_phrase(label, count) for label, count in counts.most_common())
    position_groups: dict[str, list[str]] = {"left": [], "center": [], "right": []}
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: item["center_x"]):
        key = (row["label"], row["location"])
        if key not in seen:
            position_groups[row["location"]].append(row["label"])
            seen.add(key)
    locations = []
    for position in ("left", "center", "right"):
        labels = position_groups[position]
        if labels:
            names = ", ".join(labels[:3])
            phrase = "in the center" if position == "center" else f"on the {position}"
            locations.append(f"{names} {('is' if len(labels) == 1 else 'are')} {phrase}")

    strongest = max(rows, key=lambda row: row["score"])
    text = (
        f"I see {count_text}. "
        f"By position, {'; '.join(locations)}. "
        f"The strongest detection is {strongest['label']} at {strongest['score']:.0%} confidence."
    )
    return (text, rows) if return_rows else text
